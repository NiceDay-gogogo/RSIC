#!/usr/bin/env python3
import torch.nn.functional as F
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models.FC5lstm import *
from models.AOAModel import AoADecoder
from models.AOAHFAMCFIMModel import build_aoa_hfam_cfim_models
from models.AOAAttrGCNModel import build_aoa_attr_gcn_models, load_attr_resources
from models.AOAHFAMCFIMAttrGCNModel import build_aoa_hfam_cfim_attr_gcn_models
from models.AOAHFAMAttrGCNmodel import build_aoa_hfam_attr_gcn_models
from models.AOACFIMAttrGCNModel import build_aoa_cfim_attr_gcn_models
from models.AOAMADSAP import build_aoa_mad_sap_models
from models.AOAHFAMCFIMMADSAP import build_aoa_hfam_cfim_mad_sap_models
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import codecs
import json
import numpy as np
from torch.optim.lr_scheduler import StepLR
import logging
import os
import sys
dataset = "UCM"
model = "33LFC5LSTM"
logger = logging.getLogger(__name__)


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("HCNetTrain")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger


def build_wordid_to_attrid(word_map, attr_vocab):
    mapping = torch.full((len(word_map),), -1, dtype=torch.long)
    for idx, word in enumerate(attr_vocab):
        if word in word_map:
            word_id = word_map[word]
        elif word.lower() in word_map:
            word_id = word_map[word.lower()]
        else:
            continue
        mapping[word_id] = idx
    return mapping


def compute_subsequent_labels(caps_sorted, wordid_to_attrid):
    batch_size, seq_len = caps_sorted.size()
    labels = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=caps_sorted.device)
    previous = torch.full((batch_size,), -1, dtype=torch.long, device=caps_sorted.device)
    for t in range(seq_len - 1, -1, -1):
        attr_idx = wordid_to_attrid[caps_sorted[:, t]]
        has_attr = attr_idx >= 0
        previous = torch.where(has_attr, attr_idx, previous)
        labels[:, t] = previous
    return labels

def train(args, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer,  epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    encoder.train()
    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()

    # Batches
    best_bleu4 = 0.  # BLEU-4 score right now
    steps_since_improvement = 0
    final_args = {"emb_dim": args.emb_dim,
                  "attention_dim": args.attention_dim,
                  "decoder_dim": args.decoder_dim,
                  "n_heads": args.n_heads,
                  "dropout": args.dropout,
                  "decoder_mode": args.decoder_mode,
                  "attention_method": args.attention_method,
                  "encoder_layers": args.encoder_layers,
                  "decoder_layers": args.decoder_layers,
                  "lambda_align": args.lambda_align,
                  "lambda_cap": args.lambda_cap,
                  "attr_topk": args.attr_topk,
                  "lambda_attr": args.lambda_attr,
                  "lambda_sap": args.lambda_sap,
                  "attr_alpha": args.attr_alpha,
                  "attr_gamma": args.attr_gamma,
                  "attr_warmup_epochs": args.attr_warmup_epochs,
                  "attr_ramp_epochs": args.attr_ramp_epochs}
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        attr_pos_ratio = None
        if args.decoder_mode in (
            'aoa_attr_gcn',
            'aoa_hfam_cfim_attr_gcn',
            'aoa_hfam_attr_gcn',
            'aoa_cfim_attr_gcn',
            'aoa_mad_sap',
            'aoa_hfam_cfim_mad_sap',
        ):
            imgs, caps, caplens, y_attr = batch
            y_attr = y_attr.to(device)
            attr_pos_ratio = (y_attr > 0).float().mean().item()
        else:
            imgs, caps, caplens = batch
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        # imgs: [batch_size, 14, 14, 2048]
        # caps: [batch_size, 52]
        # caplens: [batch_size, 1]
        sap_loss = torch.tensor(0., device=device)
        if args.decoder_mode in ('lstm_attention', 'aoa', 'aoa_hfam_cfim'):
            scores, caps_sorted, decode_lengths, alphas, sort_ind, img_feature, text_feature = decoder(imgs, caps, caplens)
            attr_loss = torch.tensor(0., device=device)
        elif args.decoder_mode in (
            'aoa_attr_gcn',
            'aoa_hfam_cfim_attr_gcn',
            'aoa_hfam_attr_gcn',
            'aoa_cfim_attr_gcn',
            'aoa_mad_sap',
            'aoa_hfam_cfim_mad_sap',
        ):
            outputs = decoder(imgs, caps, caplens)
            if args.decoder_mode in ('aoa_mad_sap', 'aoa_hfam_cfim_mad_sap'):
                scores, caps_sorted, decode_lengths, alphas, sort_ind, img_feature, text_feature, attr_logits, subsequent_logprobs = outputs
            else:
                scores, caps_sorted, decode_lengths, alphas, sort_ind, img_feature, text_feature, attr_logits = outputs
            targets_attr = y_attr[sort_ind]
            # 修复后的 Focal Loss：pt = p*y + (1-p)*(1-y)，调制因子 = (1-pt)^gamma
            alpha_pos = getattr(args, 'attr_alpha', 0.75)  # 正类权重（稀有类应该更高）
            gamma = getattr(args, 'attr_gamma', 2.0)
            eps = 1e-8
            prob = torch.sigmoid(attr_logits)
            # pt: 对正确类别的预测概率
            pt = prob * targets_attr + (1.0 - prob) * (1.0 - targets_attr)
            # alpha_t: 正类用 alpha_pos，负类用 (1-alpha_pos)
            alpha_t = alpha_pos * targets_attr + (1.0 - alpha_pos) * (1.0 - targets_attr)
            # focal loss = -alpha_t * (1-pt)^gamma * log(pt)
            focal_weight = (1.0 - pt).clamp(min=0.0, max=1.0) ** gamma
            attr_loss = (-alpha_t * focal_weight * torch.log(pt.clamp(min=eps))).mean()
            if args.decoder_mode in ('aoa_mad_sap', 'aoa_hfam_cfim_mad_sap'):
                labels = compute_subsequent_labels(caps_sorted, args.wordid_to_attrid)
                labels = labels[:, 1:1 + subsequent_logprobs.size(1)]
                max_len = subsequent_logprobs.size(1)
                len_mask = torch.arange(max_len, device=device).unsqueeze(0) < torch.tensor(decode_lengths, device=device).unsqueeze(1)
                sap_mask = (labels >= 0) & len_mask
                labels = labels.clamp(min=0)
                nll = -subsequent_logprobs.gather(2, labels.unsqueeze(2)).squeeze(2)
                sap_loss = (nll * sap_mask.float()).sum() / (sap_mask.float().sum() + 1e-8)
        else:
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            attr_loss = torch.tensor(0., device=device)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # print(scores.size())
        # print(targets.size())

        image_features = img_feature / img_feature.norm(dim=1, keepdim=True)
        text_features = text_feature / text_feature.norm(dim=1, keepdim=True)
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 计算余弦相似度
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(logits_per_image.shape[0], dtype=torch.long)
        #logits = logits.to(device)
        labels = labels.to(device)
        loss1 = (F.cross_entropy(logits_per_image, labels) +F.cross_entropy(logits_per_text, labels)) / 2

        # Calculate loss
        loss = criterion(scores, targets)

        # Warmup/ramp for attribute loss：前期减小 attr_loss 影响
        warmup_epochs = getattr(args, 'attr_warmup_epochs', 0)
        ramp_epochs = getattr(args, 'attr_ramp_epochs', 0)
        if epoch < warmup_epochs:
            lambda_attr_eff = 0.0
        elif ramp_epochs > 0:
            lambda_attr_eff = args.lambda_attr * min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
        else:
            lambda_attr_eff = args.lambda_attr

        total_loss = (
            args.lambda_align * loss1
            + args.lambda_cap * loss
            + lambda_attr_eff * attr_loss
            + args.lambda_sap * sap_loss
        )
        # Add doubly stochastic attention regularization
        # Second loss, mentioned in paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
        # https://arxiv.org/abs/1502.03044
        # In section 4.2.1 Doubly stochastic attention regularization: We know the weights sum to 1 at a given timestep.
        # But we also encourage the weights at a single pixel p to sum to 1 across all timesteps T.
        # This means we want the model to attend to every pixel over the course of generating the entire sequence.
        # Therefore, we want to minimize the difference between 1 and the sum of a pixel's weights across all timesteps.


        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()

        if encoder_optimizer is not None:
            encoder_optimizer.step()


        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            # print('TIME: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            logger.info("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch+0, args.epochs, i+0, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))
            if attr_pos_ratio is not None:
                logger.info(
                    "attr_debug: lambda_attr_eff={:.4f} attr_pos_ratio={:.6f}".format(
                        lambda_attr_eff, attr_pos_ratio
                    )
                )
            logger.info(
                "loss_parts: cap={:.4f} align={:.4f} attr={:.4f} sap={:.4f} total={:.4f}".format(
                    loss.item(),
                    loss1.item(),
                    attr_loss.item(),
                    sap_loss.item(),
                    total_loss.item(),
                )
            )
    return losses.avg, top5accs.avg


def validate(args, val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, batch in enumerate(val_loader):

            if args.decoder_mode in (
                'aoa_attr_gcn',
                'aoa_hfam_cfim_attr_gcn',
                'aoa_hfam_attr_gcn',
                'aoa_cfim_attr_gcn',
                'aoa_mad_sap',
                'aoa_hfam_cfim_mad_sap',
            ):
                imgs, caps, caplens, allcaps, y_attr = batch
                y_attr = y_attr.to(device)
            else:
                imgs, caps, caplens, allcaps = batch

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)

            if args.decoder_mode in ('lstm_attention', 'aoa', 'aoa_hfam_cfim'):
                scores, caps_sorted, decode_lengths, alphas, sort_ind, img_feature, text_feature = decoder(imgs, caps, caplens)
            elif args.decoder_mode in (
                'aoa_attr_gcn',
                'aoa_hfam_cfim_attr_gcn',
                'aoa_hfam_attr_gcn',
                'aoa_cfim_attr_gcn',
                'aoa_mad_sap',
                'aoa_hfam_cfim_mad_sap',
            ):
                outputs = decoder(imgs, caps, caplens)
                if args.decoder_mode in ('aoa_mad_sap', 'aoa_hfam_cfim_mad_sap'):
                    scores, caps_sorted, decode_lengths, alphas, sort_ind, img_feature, text_feature, _, _ = outputs
                else:
                    scores, caps_sorted, decode_lengths, alphas, sort_ind, img_feature, text_feature, _ = outputs
            else:
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()


            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    logger.info('Validation：')
    metrics = get_eval_score(references, hypotheses)
    logger.info("Validation summary: loss {:.4f}, top5_acc {:.2f}".format(losses.avg, top5accs.avg))
    logger.info("Metrics: Bleu_1 {Bleu_1:.4f} Bleu_2 {Bleu_2:.4f} Bleu_3 {Bleu_3:.4f} Bleu_4 {Bleu_4:.4f} "
                "METEOR {METEOR:.4f} ROUGE_L {ROUGE_L:.4f} CIDEr {CIDEr:.4f} SPICE {SPICE:.4f}".format(
                    Bleu_1=metrics.get("Bleu_1", float("nan")),
                    Bleu_2=metrics.get("Bleu_2", float("nan")),
                    Bleu_3=metrics.get("Bleu_3", float("nan")),
                    Bleu_4=metrics.get("Bleu_4", float("nan")),
                    METEOR=metrics.get("METEOR", float("nan")),
                    ROUGE_L=metrics.get("ROUGE_L", float("nan")),
                    CIDEr=metrics.get("CIDEr", float("nan")),
                    SPICE=metrics.get("SPICE", float("nan"))
                ))

    # print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
    #       (losses.avg, top5accs.avg,  metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"],
    #        metrics["METEOR"],metrics["ROUGE_L"], metrics["CIDEr"]))
    logger.info('\n')

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')

    # Data parameters
    parser.add_argument('--data_folder', default="./data/UCM_images1",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="UCM_5_cap_per_img_4_min_word_freq",help='base name shared by data files.')


    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings.')#300
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers.')
    parser.add_argument('--decoder_dim', type=int, default=1000, help='dimension of decoder RNN.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    # FIXME:note to change these
    parser.add_argument('--encoder_mode', default="resnet101", help='which model does encoder use?') # inception_v3 or vgg16 or vgg19 or resnet50 or resnet101 or resnet152
    # encoder_fusion 已在本消融实验脚本中移除：由 decoder_mode 决定是否启用 HFAM
    parser.add_argument('--decoder_mode', default="aoa", help='decoder_mode: aoa | aoa_hfam_cfim | aoa_attr_gcn | aoa_hfam_attr_gcn | aoa_cfim_attr_gcn | aoa_hfam_cfim_attr_gcn | aoa_mad_sap | aoa_hfam_cfim_mad_sap')
    parser.add_argument('--attr_dir', default=None, help='path to attr resources (attr_vocab.json, adj.npy, attr_labels.npy).')
    parser.add_argument('--lambda_attr', type=float, default=0.1, help='weight for attribute focal loss.')
    parser.add_argument('--lambda_align', type=float, default=1.0, help='weight for image-text alignment loss.')
    parser.add_argument('--lambda_cap', type=float, default=5.0, help='weight for caption cross-entropy loss.')
    parser.add_argument('--lambda_sap', type=float, default=0.2, help='weight for SAP loss in AOAMADSAP.')
    parser.add_argument('--attr_topk', type=int, default=None, help='top-k attributes to attend in GCN fusion (None=all).')
    parser.add_argument('--attr_alpha', type=float, default=0.75, help='positive class weight for focal loss (higher = more weight on rare positives).')
    parser.add_argument('--attr_gamma', type=float, default=2.0, help='gamma for focal loss modulation.')
    parser.add_argument('--attr_warmup_epochs', type=int, default=5, help='epochs with lambda_attr=0 (warmup phase).')
    parser.add_argument('--attr_ramp_epochs', type=int, default=10, help='epochs to linearly ramp lambda_attr after warmup.')

    parser.add_argument('--attention_method', default="ByPixel", help='which attention method to use?')  # ByPixel or ByChannel
    parser.add_argument('--encoder_layers', type=int, default=3, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=3, help='the number of layers of decoder in Transformer.')


    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=20, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=16, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1., help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool, default= True, help='whether fine-tune encoder or not')
    parser.add_argument('--fine_tune_embedding', type=bool, default= True, help='whether fine-tune word embeddings or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--embedding_path', default=None, help='path to pre-trained word Embedding.')
    parser.add_argument('--save_dir', default="./save/save_ucm", help='directory to store checkpoints and logs.')

    args = parser.parse_args()
    log_path = os.path.join(args.save_dir, "log_ucm.txt")
    logger = setup_logger(log_path)

    for encoder_layers, decoder_layers in [(3,3)]: #,,(0,6),(2,2),
        args.encoder_layers = encoder_layers
        args.decoder_layers = decoder_layers
        # args.encoder_mode = encoder_mode

        # load checkpoint, these parameters can't be modified
        final_args = {"emb_dim": args.emb_dim,
                     "attention_dim": args.attention_dim,
                     "decoder_dim": args.decoder_dim,
                     "n_heads": args.n_heads,
                     "dropout": args.dropout,
                     "decoder_mode": args.decoder_mode,
                     "attention_method": args.attention_method,
                     "encoder_layers": args.encoder_layers,
                     "decoder_layers": args.decoder_layers,
                     "lambda_align": args.lambda_align,
                     "lambda_cap": args.lambda_cap,
                     "attr_topk": args.attr_topk,
                     "lambda_attr": args.lambda_attr,
                     "lambda_sap": args.lambda_sap,
                     "attr_alpha": args.attr_alpha,
                     "attr_gamma": args.attr_gamma,
                     "attr_warmup_epochs": args.attr_warmup_epochs,
                     "attr_ramp_epochs": args.attr_ramp_epochs}

        start_epoch = 0
        best_bleu4 = 0.  # BLEU-4 score right now
        epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
        cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
        # print(device)

        # Read word map
        word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        attr_dir = args.attr_dir or os.path.join(args.data_folder, "attr")
        attr_vocab, _ = load_attr_resources(attr_dir)
        args.wordid_to_attrid = build_wordid_to_attrid(word_map, attr_vocab).to(device)

        def build_models():
            if args.decoder_mode == "aoa":
                encoder = Encoder(NetType=args.encoder_mode)
                decoder = AoADecoder(
                    attention_dim=args.attention_dim,
                    embed_dim=args.emb_dim,
                    decoder_dim=args.decoder_dim,
                    vocab_size=len(word_map),
                    encoder_dim=1024,
                    num_heads=args.n_heads,
                    dropout=args.dropout,
                )
            elif args.decoder_mode == "aoa_hfam_cfim":
                encoder, decoder = build_aoa_hfam_cfim_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                )
            elif args.decoder_mode == "aoa_attr_gcn":
                encoder, decoder = build_aoa_attr_gcn_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    attr_dir=attr_dir,
                    attr_topk=args.attr_topk,
                )
            elif args.decoder_mode == "aoa_hfam_attr_gcn":
                encoder, decoder = build_aoa_hfam_attr_gcn_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    attr_dir=attr_dir,
                    attr_topk=args.attr_topk,
                )
            elif args.decoder_mode == "aoa_cfim_attr_gcn":
                encoder, decoder = build_aoa_cfim_attr_gcn_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    attr_dir=attr_dir,
                    attr_topk=args.attr_topk,
                )
            elif args.decoder_mode == "aoa_hfam_cfim_attr_gcn":
                encoder, decoder = build_aoa_hfam_cfim_attr_gcn_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    attr_dir=attr_dir,
                    attr_topk=args.attr_topk,
                )
            elif args.decoder_mode == "aoa_mad_sap":
                encoder, decoder = build_aoa_mad_sap_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    attr_dir=attr_dir,
                    attr_topk=args.attr_topk,
                    word_map=word_map,
                )
            elif args.decoder_mode == "aoa_hfam_cfim_mad_sap":
                encoder, decoder = build_aoa_hfam_cfim_mad_sap_models(
                    vocab_size=len(word_map),
                    embed_dim=args.emb_dim,
                    attention_dim=args.attention_dim,
                    decoder_dim=args.decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    attr_dir=attr_dir,
                    attr_topk=args.attr_topk,
                    word_map=word_map,
                )
            else:
                raise ValueError(f"Unsupported decoder_mode: {args.decoder_mode}")
            return encoder, decoder

        # Initialize / load checkpoint
        if args.checkpoint is None:
            encoder, decoder = build_models()

            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = (
                torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)
                if args.fine_tune_encoder
                else None
            )
            encoder_lr_scheduler = StepLR(encoder_optimizer, step_size=600, gamma=0.9) if encoder_optimizer is not None else None

            decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                 lr=args.decoder_lr)
            decoder_lr_scheduler = StepLR(decoder_optimizer,step_size=600,gamma=0.9)

            # load pre-trained word embedding
            if args.embedding_path is not None:
                all_word_embeds = {}
                for i, line in enumerate(codecs.open(args.embedding_path, 'r', 'utf-8')):
                    s = line.strip().split()
                    all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

                # change emb_dim
                args.emb_dim = list(all_word_embeds.values())[-1].size
                word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_map), args.emb_dim))
                for w in word_map:
                    if w in all_word_embeds:
                        word_embeds[word_map[w]] = all_word_embeds[w]
                    elif w.lower() in all_word_embeds:
                        word_embeds[word_map[w]] = all_word_embeds[w.lower()]
                    else:
                        # <pad> <start> <end> <unk>
                        embedding_i = torch.ones(1, args.emb_dim)
                        torch.nn.init.xavier_uniform_(embedding_i)
                        word_embeds[word_map[w]] = embedding_i

                word_embeds = torch.FloatTensor(word_embeds).to(device)
                decoder.load_pretrained_embeddings(word_embeds)
                decoder.fine_tune_embeddings(args.fine_tune_embedding)
                logger.info('Loaded {} pre-trained word embeddings.'.format(len(word_embeds)))

        else:
            logger.info("isNone")
            logger.info(args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=str(device))
            start_epoch = checkpoint.get('epoch', 0) + 1
            epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)

            if 'encoder' in checkpoint and 'decoder' in checkpoint:
                encoder = checkpoint['encoder']
                encoder_optimizer = checkpoint.get('encoder_optimizer')
                decoder = checkpoint['decoder']
                decoder_optimizer = checkpoint.get('decoder_optimizer')
                decoder.fine_tune_embeddings(args.fine_tune_embedding)
            else:
                encoder, decoder = build_models()
                encoder.fine_tune(args.fine_tune_encoder)
                encoder_optimizer = (
                    torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)
                    if args.fine_tune_encoder
                    else None
                )
                decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                     lr=args.decoder_lr)
                encoder_state = checkpoint.get('encoder_state_dict')
                decoder_state = checkpoint.get('decoder_state_dict')
                if encoder_state is not None:
                    encoder.load_state_dict(encoder_state)
                if decoder_state is not None:
                    decoder.load_state_dict(decoder_state)
                ckpt_dir = os.path.dirname(args.checkpoint)
                decoder_opt_path = os.path.join(ckpt_dir, "optimizer.pth")
                encoder_opt_path = os.path.join(ckpt_dir, "encoder_optimizer.pth")
                if "best" in os.path.basename(args.checkpoint).lower():
                    if not os.path.exists(decoder_opt_path):
                        decoder_opt_path = os.path.join(ckpt_dir, "best_optimizer.pth")
                    if not os.path.exists(encoder_opt_path):
                        encoder_opt_path = os.path.join(ckpt_dir, "best_encoder_optimizer.pth")
                if decoder_optimizer is not None and os.path.exists(decoder_opt_path):
                    decoder_optimizer.load_state_dict(torch.load(decoder_opt_path, map_location=str(device)))
                if encoder_optimizer is not None and os.path.exists(encoder_opt_path):
                    encoder_optimizer.load_state_dict(torch.load(encoder_opt_path, map_location=str(device)))

            # load final_args from checkpoint
            final_args = checkpoint.get('final_args', final_args)
            for key in final_args.keys():
                args.__setattr__(key, final_args[key])
            if args.fine_tune_encoder is True and encoder_optimizer is None:
                logger.info("Encoder_Optimizer is None, Creating new Encoder_Optimizer!")
                encoder.fine_tune(args.fine_tune_encoder)
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                     lr=args.encoder_lr)

        # Move to GPU, if available
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        logger.info("Encoder_mode:{}   Decoder_mode:{}".format(args.encoder_mode, args.decoder_mode))
        logger.info("encoder_layers {} decoder_layers {} n_heads {} dropout {} attention_method {} encoder_lr {} "
                    "decoder_lr {} alpha_c {}".format(args.encoder_layers, args.decoder_layers, args.n_heads, args.dropout,
                                                      args.attention_method, args.encoder_lr, args.decoder_lr, args.alpha_c))
        # print(encoder)
        # print(decoder)

        # Loss function
        criterion = nn.CrossEntropyLoss().to(device)

        # Custom dataloaders
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.399, 0.410, 0.371], std=[0.151, 0.138, 0.134])
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        load_attr_flag = True if args.decoder_mode in (
            'aoa_attr_gcn',
            'aoa_hfam_cfim_attr_gcn',
            'aoa_hfam_attr_gcn',
            'aoa_cfim_attr_gcn',
            'aoa_mad_sap',
            'aoa_hfam_cfim_mad_sap',
        ) else False

        # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
        train_loader = torch.utils.data.DataLoader(
            CaptionDataset(args.data_folder, args.data_name, 'TRAIN',
                           transform=transforms.Compose([transforms.RandomHorizontalFlip(),normalize]),
                           load_attr=load_attr_flag),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CaptionDataset(args.data_folder, args.data_name, 'VAL',
                           transform=transforms.Compose([transforms.RandomHorizontalFlip(),normalize]),
                           load_attr=load_attr_flag),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        metrics_history = []
        infos = {
            "epoch": int(start_epoch),
            "epochs_since_improvement": int(epochs_since_improvement),
            "final_args": final_args,
            "decoder_mode": args.decoder_mode,
            "encoder_mode": args.encoder_mode,
        }
        histories = {"metrics_history": metrics_history}

        # Epochs
        for epoch in range(start_epoch, args.epochs):

            # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 25
            # 8 20
            if epochs_since_improvement == args.stop_criteria:
                logger.info("the model has not improved in the last {} epochs".format(args.stop_criteria))
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if args.fine_tune_encoder and encoder_optimizer is not None:
                    logger.info(encoder_optimizer)
                    adjust_learning_rate(encoder_optimizer, 0.8)

            # One epoch's training
            train_loss, train_top5 = train(args,
                                           train_loader=train_loader,
                                           # val_loader=val_loader,
                                           encoder=encoder,
                                           decoder=decoder,
                                           criterion=criterion,
                                           encoder_optimizer=encoder_optimizer,
                                           #encoder_lr_scheduler=encoder_lr_scheduler,
                                           decoder_optimizer=decoder_optimizer,
                                           #decoder_lr_scheduler=decoder_lr_scheduler,
                                           epoch=epoch)


            # One epoch's validation
            metrics = validate(args,
                               val_loader=val_loader,
                               encoder=encoder,
                               decoder=decoder,
                               criterion=criterion)
            logger.info("Epoch {} training summary: loss {:.4f}, top5_acc {:.2f}".format(epoch, train_loss, train_top5))

            recent_bleu4 = metrics["Bleu_4"]

            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            checkpoint_name = model+"_"+dataset #_tengxun_aggregation
            save_checkpoint(checkpoint_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                            decoder_optimizer, metrics, is_best, final_args,
                            output_dir=args.save_dir, current_name="model.pth", best_name="best_model.pth",
                            infos=infos, histories=histories)

            metrics_clean = {k: float(v) for k, v in metrics.items()}
            metrics_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_top5": float(train_top5),
                    "metrics": metrics_clean,
                }
            )
            infos["epoch"] = int(epoch)
            infos["epochs_since_improvement"] = int(epochs_since_improvement)
            history_path = os.path.join(args.save_dir, "metrics_history.json")
            with open(history_path, "w") as f:
                json.dump(metrics_history, f, indent=2)
