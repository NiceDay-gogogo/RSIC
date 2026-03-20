import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.AOAModel import MultiHeadedDotAttention
from models.AOAMADSAP import MADDetector, SAPSelector, AttrAttention
from models.AOAAttrGCNModel import load_attr_resources
from models.AOAHFAMCFIMModel import HFAMV2Encoder, CFIM
from models.FC5lstm import TextEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AoAHFAMCFIMMADSAPDecoder(nn.Module):
    """AoA + HFAM + CFIM + MAD + SAP decoder."""

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim,
        num_heads,
        multi_head_scale,
        dropout,
        dropout_aoa,
        attr_size,
        attr_vocab,
        transition_matrix,
        selected_num=10,
        word_map=None,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.multi_head_scale = multi_head_scale
        self.attr_size = attr_size
        self.attr_vocab = attr_vocab

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mha = MultiHeadedDotAttention(
            num_heads,
            decoder_dim,
            dropout=dropout,
            scale=multi_head_scale,
            project_k_v=1,
            do_aoa=1,
            norm_q=1,
            dropout_aoa=dropout_aoa,
        )

        self.att_lstm = nn.LSTMCell(embed_dim + encoder_dim + decoder_dim, decoder_dim)
        self.language_lstm = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim)
        self.ctx_gate = nn.Sequential(nn.Linear(decoder_dim * multi_head_scale + decoder_dim, 2 * decoder_dim), nn.GLU())

        self.cfim = CFIM(encoder_dim=encoder_dim, embed_dim=embed_dim, attention_dim=attention_dim)

        self.mad = MADDetector(encoder_dim=encoder_dim, attr_size=attr_size, attr_embed_dim=embed_dim, mad_dim=decoder_dim // 2)
        self.attr_embed_proj = nn.Linear(embed_dim, decoder_dim)
        self.sap = SAPSelector(
            attr_size=attr_size,
            decoder_dim=decoder_dim,
            selected_num=selected_num,
            transition_matrix=transition_matrix,
        )
        self.attr_attention = AttrAttention(decoder_dim=decoder_dim, attention_dim=attention_dim)
        self.attr_gate = nn.Sequential(nn.Linear(decoder_dim * 2 + decoder_dim, decoder_dim), nn.Sigmoid())

        self.init_h = nn.Linear(decoder_dim, decoder_dim)
        self.init_c = nn.Linear(decoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.textencoder = TextEncoder(input_size=embed_dim, hidden_size=decoder_dim, output_size=attention_dim)
        self.img_proj = nn.Linear(encoder_dim, attention_dim)

        self._build_word_to_attr(word_map)
        self.init_weights()

    def _build_word_to_attr(self, word_map):
        if word_map is None:
            self.register_buffer("wordid_to_attrid", torch.full((self.vocab_size,), -1, dtype=torch.long))
            return
        mapping = torch.full((self.vocab_size,), -1, dtype=torch.long)
        for idx, word in enumerate(self.attr_vocab):
            if word in word_map:
                word_id = word_map[word]
            elif word.lower() in word_map:
                word_id = word_map[word.lower()]
            else:
                continue
            mapping[word_id] = idx
        self.register_buffer("wordid_to_attrid", mapping)

    def init_attr_embeddings(self, word_map):
        if word_map is None:
            return
        with torch.no_grad():
            for idx, word in enumerate(self.attr_vocab):
                if word in word_map:
                    word_id = word_map[word]
                elif word.lower() in word_map:
                    word_id = word_map[word.lower()]
                else:
                    continue
                self.mad.attr_embed.weight[idx].copy_(self.embedding.weight[word_id])

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, mean_feats):
        h = self.init_h(mean_feats)
        c = self.init_c(mean_feats)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        proj_feats = self.encoder_proj(encoder_out)
        mean_feats = proj_feats.mean(dim=1)

        embeddings = self.dropout(self.embedding(encoded_captions))
        text_feature = self.textencoder(embeddings.clone())

        h_att, c_att = self.init_hidden_state(mean_feats)
        h_lang, c_lang = self.init_hidden_state(mean_feats)

        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_len = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, num_pixels, device=encoder_out.device)
        subsequent_logprobs = torch.zeros(batch_size, max_decode_len, self.attr_size, device=encoder_out.device)

        attr_probs, attr_logits = self.mad(encoder_out)
        attr_emb = self.attr_embed_proj(self.mad.attr_embed.weight)

        previous_attr = torch.zeros(batch_size, dtype=torch.long, device=encoder_out.device)
        prev_has_attr = torch.zeros(batch_size, dtype=torch.bool, device=encoder_out.device)

        for t in range(max_decode_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            xt = embeddings[:batch_size_t, t, :]
            h_prev = h_att[:batch_size_t]
            c_prev = c_att[:batch_size_t]
            h_lang_prev = h_lang[:batch_size_t]

            word_ids = encoded_captions[:batch_size_t, t]
            attr_idx = self.wordid_to_attrid[word_ids]
            has_attr = attr_idx >= 0
            prev_attr_t = torch.where(has_attr, attr_idx, previous_attr[:batch_size_t])
            prev_has_t = prev_has_attr[:batch_size_t] | has_attr
            with torch.no_grad():
                previous_attr[:batch_size_t] = prev_attr_t
                prev_has_attr[:batch_size_t] = prev_has_t

            cfim_ctx = self.cfim(h_lang_prev, xt, encoder_out[:batch_size_t])
            h_new, c_new = self.att_lstm(
                torch.cat([xt, cfim_ctx, h_lang_prev], dim=1), (h_prev, c_prev)
            )

            att_ctx = self.mha(h_new, proj_feats[:batch_size_t], proj_feats[:batch_size_t], None).squeeze(1)
            merged_ctx = self.ctx_gate(torch.cat([att_ctx, h_new], dim=1))

            subsequent_prob, selected_attr_emb = self.sap(
                attr_emb,
                attr_probs[:batch_size_t],
                h_new,
                prev_attr_t,
                prev_has_t.float(),
            )
            subsequent_logprobs[:batch_size_t, t, :] = subsequent_prob
            attr_ctx = self.attr_attention(h_new, selected_attr_emb)
            gate = self.attr_gate(torch.cat([merged_ctx, attr_ctx, h_new], dim=1))
            fused_ctx = gate * merged_ctx + (1 - gate) * attr_ctx

            h_lang_new, c_lang_new = self.language_lstm(
                torch.cat([h_new, fused_ctx], dim=1),
                (h_lang_prev, c_lang[:batch_size_t]),
            )

            preds = self.fc(self.dropout(h_lang_new))
            predictions[:batch_size_t, t, :] = preds

            if self.mha.attn is not None:
                alpha = self.mha.attn.mean(dim=1).squeeze(1)
            else:
                alpha = torch.full((batch_size_t, num_pixels), 1.0 / num_pixels, device=proj_feats.device)
            alphas[:batch_size_t, t, :] = alpha

            if batch_size_t < h_att.size(0):
                h_att = torch.cat([h_new, h_att[batch_size_t:]], dim=0)
                c_att = torch.cat([c_new, c_att[batch_size_t:]], dim=0)
                h_lang = torch.cat([h_lang_new, h_lang[batch_size_t:]], dim=0)
                c_lang = torch.cat([c_lang_new, c_lang[batch_size_t:]], dim=0)
            else:
                h_att = h_new
                c_att = c_new
                h_lang = h_lang_new
                c_lang = c_lang_new

        img_feature = self.img_proj(encoder_out.mean(1)).squeeze(1)
        return (
            predictions,
            encoded_captions,
            decode_lengths,
            alphas,
            sort_ind,
            img_feature,
            text_feature,
            attr_logits,
            subsequent_logprobs,
        )


def build_aoa_hfam_cfim_mad_sap_models(
    vocab_size,
    embed_dim,
    attention_dim,
    decoder_dim,
    encoder_backbone="resnet101",
    n_heads=8,
    dropout=0.5,
    attr_dir="./data/UCM/attr",
    attr_topk=10,
    word_map=None,
):
    vocab, adj = load_attr_resources(attr_dir)
    attr_size = len(vocab)

    markov_path = os.path.join(attr_dir, "markov_mat.npy")
    if os.path.exists(markov_path):
        trans = torch.tensor(np.load(markov_path)).float()
    else:
        trans = adj.clone()
    if trans.dim() == 2:
        trans = trans + torch.eye(attr_size)
        row_sum = trans.sum(dim=1, keepdim=True).clamp(min=1e-6)
        trans = trans / row_sum

    encoder = HFAMV2Encoder(NetType=encoder_backbone)
    decoder = AoAHFAMCFIMMADSAPDecoder(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=1024,
        num_heads=n_heads,
        multi_head_scale=1,
        dropout=dropout,
        dropout_aoa=0.3,
        attr_size=attr_size,
        attr_vocab=vocab,
        transition_matrix=trans,
        selected_num=attr_topk or 10,
        word_map=word_map,
    )
    decoder.init_attr_embeddings(word_map)
    return encoder, decoder
