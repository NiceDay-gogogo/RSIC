from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AoAModel import AoAModel


class AoAAllAttrProbModel(AoAModel):
    """
    AoA model variant that consumes probabilistic attribute vectors at every decoding step.

    Pipeline:
      1. Attribute probabilities from JSON/loader are thresholded at 0.7 and
         L1-normalized per sample.
      2. The normalized vector is embedded twice:
         - one embedding matches rnn_size and is fused with mean_feats (t=0 inject prior).
         - another embedding matches input_encoding_size and is concatenated with each word embedding at every step.
    """

    def __init__(self, opt):
        super(AoAAllAttrProbModel, self).__init__(opt)
        self.attr_feat_size = getattr(opt, "attr_feat_size", 40)
        self.attr_prob_threshold = getattr(opt, "attr_prob_threshold", 0.7)
        self.attr_eps = getattr(opt, "attr_norm_eps", 1e-6)

        # Embedding to inject into mean features (rnn_size dimension)
        self.attr_mean_embed = nn.Sequential(
            nn.Linear(self.attr_feat_size, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm),
        )
        self.attr_mean_proj = nn.Linear(self.rnn_size * 2, self.rnn_size)

        # Embedding to fuse with word embeddings (input_encoding_size dimension)
        self.attr_word_embed = nn.Sequential(
            nn.Linear(self.attr_feat_size, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm),
        )
        self.word_attr_fusion = nn.Sequential(
            nn.Linear(self.input_encoding_size * 2, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm),
        )

        self._cached_attr_probs = None
        self._cached_word_attr_emb = None
        self._cached_word_attr_emb_full = None

    # ------------------------------------------------------------------
    def _process_attr_probs(self, attr_labels, ref_tensor):
        if attr_labels is None:
            attr = ref_tensor.new_zeros(ref_tensor.size(0), self.attr_feat_size)
        else:
            attr = attr_labels.to(ref_tensor.device).float()

        attr = torch.where(attr >= self.attr_prob_threshold, attr, torch.zeros_like(attr))
        sums = attr.sum(dim=1, keepdim=True) + self.attr_eps
        attr = attr / sums
        return attr

    def _inject_attr_into_mean(self, mean_feats, attr_probs):
        attr_emb = self.attr_mean_embed(attr_probs)
        fused = torch.cat([mean_feats, attr_emb], dim=1)
        return self.attr_mean_proj(fused)

    def _prepare_attr_word_emb(self, attr_probs):
        return self.attr_word_embed(attr_probs)

    # ------------------------------------------------------------------
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        mean_feats, att_feats, p_att_feats, att_masks = super(AoAAllAttrProbModel, self)._prepare_feature(
            fc_feats, att_feats, att_masks
        )

        attr_probs = self._cached_attr_probs
        if attr_probs is None:
            attr_probs = mean_feats.new_zeros(mean_feats.size(0), self.attr_feat_size)
        mean_feats = self._inject_attr_into_mean(mean_feats, attr_probs)
        word_attr_emb = self._prepare_attr_word_emb(attr_probs)
        self._cached_word_attr_emb = word_attr_emb
        self._cached_word_attr_emb_full = word_attr_emb

        return mean_feats, att_feats, p_att_feats, att_masks

    # ------------------------------------------------------------------
    def _forward(self, fc_feats, att_feats, seq, att_masks=None, attr_labels=None):
        attr_processed = self._process_attr_probs(attr_labels, fc_feats)
        self._cached_attr_probs = attr_processed
        try:
            return super(AoAAllAttrProbModel, self)._forward(fc_feats, att_feats, seq, att_masks)
        finally:
            self._cached_attr_probs = None
            self._cached_word_attr_emb = None
            self._cached_word_attr_emb_full = None

    def _sample(self, fc_feats, att_feats, att_masks=None, attr_labels=None, opt={}):
        attr_processed = self._process_attr_probs(attr_labels, fc_feats)
        self._cached_attr_probs = attr_processed
        try:
            return super(AoAAllAttrProbModel, self)._sample(fc_feats, att_feats, att_masks, opt)
        finally:
            self._cached_attr_probs = None
            self._cached_word_attr_emb = None
            self._cached_word_attr_emb_full = None

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_mean_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        attr_word_emb_full = self._cached_word_attr_emb_full

        assert beam_size <= self.vocab_size + 1
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]

        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_mean = p_mean_feats[k:k+1].expand(beam_size, p_mean_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            if p_att_masks is not None:
                tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,) + p_att_masks.size()[1:])).contiguous()
            else:
                tmp_att_masks = None

            if attr_word_emb_full is not None:
                tmp_attr = attr_word_emb_full[k:k+1].expand(beam_size, -1).contiguous()
            else:
                tmp_attr = tmp_mean.new_zeros(beam_size, self.input_encoding_size)
            self._cached_word_attr_emb = tmp_attr

            for t in range(1):
                if t == 0:
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)
                logprobs, state = self.get_logprobs_state(it, tmp_mean, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_mean, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        self._cached_attr_probs = None
        self._cached_word_attr_emb = None
        self._cached_word_attr_emb_full = None
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    # ------------------------------------------------------------------
    def get_logprobs_state(self, it, mean_feats, att_feats, p_att_feats, att_masks, state):
        word_emb = self.embed(it)
        attr_emb = self._cached_word_attr_emb
        if attr_emb is None or attr_emb.size(0) != word_emb.size(0):
            attr_emb = word_emb.new_zeros(word_emb.size(0), self.input_encoding_size)
        xt = self.word_attr_fusion(torch.cat([word_emb, attr_emb], dim=1))
        output, state = self.core(xt, mean_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state
