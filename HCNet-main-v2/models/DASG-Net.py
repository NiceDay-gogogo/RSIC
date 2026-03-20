import torch
from torch import nn

from models.AOAModel import AoADecoder
from models.AOAAttrGCNModel import SimpleGCN, NoisyORPrior, load_attr_resources
from models.AOAHFAMCFIMModel import HFAMV2Encoder, CFIM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AoAHFAMCFIMAttrGCNDecoder(nn.Module):
    """AoA + CFIM + Attr(GCN) decoder. Used with HFAMV2Encoder."""

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
        adj_matrix,
        attr_topk=None,
    ):
        super().__init__()
        self.base = AoADecoder(
            attention_dim,
            embed_dim,
            decoder_dim,
            vocab_size,
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            multi_head_scale=multi_head_scale,
            dropout=dropout,
            dropout_aoa=dropout_aoa,
        )

        self.cfim = CFIM(encoder_dim=encoder_dim, embed_dim=embed_dim, attention_dim=attention_dim)
        self.att_lstm = nn.LSTMCell(embed_dim + encoder_dim + decoder_dim, decoder_dim)

        self.attr_prior = NoisyORPrior(encoder_dim=encoder_dim, attr_dim=decoder_dim // 2, attr_size=attr_size)
        self.attr_classifier_dyn = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_dim, attr_size),
        )
        self.gcn = SimpleGCN(
            attr_size,
            in_dim=decoder_dim // 2,
            hidden_dim=decoder_dim,
            out_dim=decoder_dim,
            adj_matrix=adj_matrix,
        )
        self.fuse_gate = nn.Sequential(nn.Linear(decoder_dim * 2 + decoder_dim, decoder_dim), nn.Sigmoid())
        self.attr_topk = attr_topk

    @property
    def vocab_size(self):
        return self.base.vocab_size

    def load_pretrained_embeddings(self, embeddings):
        return self.base.load_pretrained_embeddings(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        return self.base.fine_tune_embeddings(fine_tune)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.base.encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        proj_feats = self.base.encoder_proj(encoder_out)
        mean_feats = proj_feats.mean(dim=1)
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.base.dropout(self.base.embedding(encoded_captions))
        text_feature = self.base.textencoder(embeddings.clone())

        h_att, c_att = self.base.init_hidden_state(mean_feats)
        h_lang, c_lang = self.base.init_hidden_state(mean_feats)

        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), self.base.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels, device=encoder_out.device)

        gcn_feats = self.gcn()
        attr_logits_img = self.attr_prior(encoder_out)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            xt = embeddings[:batch_size_t, t, :]
            h_prev = h_att[:batch_size_t]
            c_prev = c_att[:batch_size_t]
            h_lang_prev = h_lang[:batch_size_t]

            # CFIM injection (use encoder_out to match HCNet CFIM)
            cfim_ctx = self.cfim(h_lang_prev, xt, encoder_out[:batch_size_t])
            att_input = torch.cat([xt, cfim_ctx, h_lang_prev], dim=1)
            h_new, c_new = self.att_lstm(att_input, (h_prev, c_prev))

            att_ctx = self.base.mha(h_new, proj_feats[:batch_size_t], proj_feats[:batch_size_t], None).squeeze(1)
            merged_ctx = self.base.ctx_gate(torch.cat([att_ctx, h_new], dim=1))

            dyn_logits = self.attr_classifier_dyn(torch.cat([att_ctx, h_new], dim=1))
            fused_logits = dyn_logits + 0.3 * attr_logits_img[:batch_size_t]
            attr_probs = torch.sigmoid(fused_logits)

            if self.attr_topk is not None and self.attr_topk > 0:
                k = min(self.attr_topk, attr_probs.size(1))
                topv, topi = torch.topk(attr_probs, k=k, dim=1)
                mask = torch.zeros_like(attr_probs)
                mask.scatter_(1, topi, topv)
                weights = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                weights = attr_probs / (attr_probs.sum(dim=1, keepdim=True) + 1e-8)

            gcn_ctx = torch.matmul(
                weights.unsqueeze(1),
                gcn_feats.unsqueeze(0).expand(batch_size_t, -1, -1),
            ).squeeze(1)

            gate = self.fuse_gate(torch.cat([merged_ctx, gcn_ctx, h_new], dim=1))
            fused_ctx = gate * merged_ctx + (1 - gate) * gcn_ctx

            h_lang_new, c_lang_new = self.base.language_lstm(
                torch.cat([h_new, fused_ctx], dim=1), (h_lang_prev, c_lang[:batch_size_t])
            )

            preds = self.base.fc(self.base.dropout(h_lang_new))
            predictions[:batch_size_t, t, :] = preds

            if self.base.mha.attn is not None:
                alpha = self.base.mha.attn.mean(dim=1).squeeze(1)
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

        img_feature = self.base.img_proj(encoder_out.mean(1)).squeeze(1)
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind, img_feature, text_feature, attr_logits_img


def build_aoa_hfam_cfim_attr_gcn_models(
    vocab_size,
    embed_dim,
    attention_dim,
    decoder_dim,
    encoder_backbone="resnet101",
    n_heads=8,
    dropout=0.5,
    attr_dir="./data/UCM/attr",
    attr_topk=None,
):
    vocab, adj = load_attr_resources(attr_dir)
    attr_size = len(vocab)

    encoder = HFAMV2Encoder(NetType=encoder_backbone)
    decoder = AoAHFAMCFIMAttrGCNDecoder(
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
        adj_matrix=adj,
        attr_topk=attr_topk,
    )
    return encoder, decoder
