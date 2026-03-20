#!/usr/bin/env python3

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
import os
from models.FC5lstm import Encoder
from models.AOAModel import AoADecoder
from models.AOAHFAMCFIMModel import build_aoa_hfam_cfim_models
from models.AOAAttrGCNModel import build_aoa_attr_gcn_models
from models.AOAHFAMAttrGCNmodel import build_aoa_hfam_attr_gcn_models
from models.AOACFIMAttrGCNModel import build_aoa_cfim_attr_gcn_models
from models.AOAHFAMCFIMAttrGCNModel import build_aoa_hfam_cfim_attr_gcn_models
from models.AOAMADSAP import build_aoa_mad_sap_models
from models.AOAHFAMCFIMMADSAP import build_aoa_hfam_cfim_mad_sap_models
# import transformer, models


def load_checkpoint_with_fallback(checkpoint_path, map_location, cpu_fallback=False):
    """
    Load a checkpoint with an optional CPU fallback when GPU memory is insufficient.
    """
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    # torch.cuda.OutOfMemoryError does not exist in older torch versions; fall back to RuntimeError
    except RuntimeError as e:
        # Retry on CPU if GPU loading runs out of memory and fallback is allowed
        if cpu_fallback and "out of memory" in str(e).lower() and map_location != "cpu":
            print("OOM loading checkpoint on {}, retrying with map_location='cpu'...".format(map_location))
            return torch.load(checkpoint_path, map_location="cpu")
        raise


def ids_to_words(seq, rev_word_map):
    special = {'<start>', '<end>', '<pad>'}
    return [rev_word_map[idx] for idx in seq if rev_word_map.get(idx, '') not in special]


def evaluate_transformer(args, rev_word_map):
    """
    Evaluation for decoder_mode: transformer

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TEST', transform=transforms.Compose([transforms.RandomHorizontalFlip(),normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (image, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            # if i>30:
            #     break
            if (i+1)%5 != 0:
                continue
            k = beam_size
            # Move to GPU device, if available
            image = image.to(device)  # [1, 3, 256, 256]

            # Encode
            encoder_out = encoder(image)  # [1, enc_image_size=14, enc_image_size=14, encoder_dim=2048]
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(-1)
            # We'll treat the problem as having a batch size of k, where k is beam_size
            encoder_out = encoder_out.expand(k, enc_image_size, enc_image_size, encoder_dim)  # [k, enc_image_size, enc_image_size, encoder_dim]
            # Tensor to store top k previous words at each step; now they're just <start>
            # Important: [1, 52] (eg: [[<start> <start> <start> ...]]) will not work, since it contains the position encoding
            k_prev_words = torch.LongTensor([[word_map['<start>']]*52] * k).to(device)  # (k, 52)
            # Tensor to store top k sequences; now they're just <start>
            seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            # Start decoding
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                # print("steps {} k_prev_words: {}".format(step, k_prev_words))
                # cap_len = torch.LongTensor([52]).repeat(k, 1).to(device) may cause different sorted results on GPU/CPU in transformer.py
                cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
                dec_out = decoder(encoder_out, k_prev_words, cap_len)
                if len(dec_out) == 7:
                    scores, _, _, _, _, _, _ = dec_out
                elif len(dec_out) == 8:
                    scores, _, _, _, _, _, _, _ = dec_out
                elif len(dec_out) == 9:
                    scores, _, _, _, _, _, _, _, _ = dec_out
                else:
                    raise ValueError(f"Unexpected decoder output length {len(dec_out)}")
                scores = scores[:, step-1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='trunc')  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # Important: this will not work, since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step+1] = seqs  # [s, 52]
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            # choose the caption which has the best_score.
            if len(complete_seqs_scores) == 0:
                # fallback: take best partial seq if no <end> found
                top_k_scores, top_k_words = scores.view(-1).topk(1, 0, True, True)
                prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='trunc')
                seq = seqs[prev_word_inds].tolist()[0]
            else:
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[indices]
            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
            # Hypotheses
            # tmp_hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            assert len(references) == len(hypotheses)
            # Print References, Hypotheses and metrics every step
            # words = []
            # # print('*' * 10 + 'ImageCaptions' + '*' * 10, len(img_captions))
            # for seq in img_captions:
            #     words.append([rev_word_map[ind] for ind in seq])
            # for i, seq in enumerate(words):
            #     print('Reference{}: '.format(i), seq)
            # print('Hypotheses: ', [rev_word_map[ind] for ind in tmp_hyp])
            # metrics = get_eval_score([img_captions], [tmp_hyp])
            # print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
            #       (args.decoder_mode, args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
            #        metrics["Bleu_4"],
            #        metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'HCNet_UCM.json'), 'w') as file:
        json.dump(hypotheses, file)
    # convert ids to words for text metrics (SPICE/METEOR expect real tokens)
    ref_words = [[ids_to_words(cap, rev_word_map) for cap in caps] for caps in references]
    hyp_words = [ids_to_words(h, rev_word_map) for h in hypotheses]
    metrics = get_eval_score(ref_words, hyp_words)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--data_folder', default="./data/UCM_images1",help=''
                                                                              ''
                                                                              'folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="UCM_5_cap_per_img_4_min_word_freq",help='base name shared by data files.')

    # FIXME:note to change these
    parser.add_argument('--encoder_mode', default="resnet101", help='which model does encoder use?') # inception_v3 or vgg16 or vgg19 or resnet50 or resnet101 or resnet152
    parser.add_argument('--decoder_mode', default="lstm_attention", help='which model does decoder use?')  # lstm or lstm_attention or transformer or transformer_decoder

    parser.add_argument('--beam_size', type=int, default=3, help='beam_size.')
    parser.add_argument('--emb_dim', type=int, default=512, help='embedding dim (used if checkpoint lacks final_args).')
    parser.add_argument('--attention_dim', type=int, default=512, help='attention dim (used if checkpoint lacks final_args).')
    parser.add_argument('--decoder_dim', type=int, default=512, help='decoder dim (used if checkpoint lacks final_args).')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads (used if checkpoint lacks final_args).')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout (used if checkpoint lacks final_args).')
    parser.add_argument('--attr_topk', type=int, default=None, help='top-k attributes (used if needed).')
    parser.add_argument('--path', default="./best_models_weights/", help='directory containing checkpoints (used if --checkpoint is not provided).')
    parser.add_argument('--checkpoint', default=None, help='path to model checkpoint (.pth or .pth.tar).')
    parser.add_argument('--metrics_out', default=None,
                        help='optional output txt path for saving metrics; defaults to results/<checkpoint_basename>_metrics.txt')
    parser.add_argument('--device', default=None, help="device to run on, e.g. 'cuda:0' or 'cpu'; defaults to cuda if available.")
    parser.add_argument('--load_on_cpu', action='store_true', help="load checkpoint on CPU to avoid GPU OOM, then move to device.")
    parser.add_argument('--load_cpu_fallback', action='store_true', help="if loading on GPU OOMs, retry loading on CPU automatically.")
    args = parser.parse_args()

    for encoder_layers, decoder_layers in [(3, 3)]:  # ,,(0,6),(2,2),


        args.encoder_layers = encoder_layers
        args.decoder_layers = decoder_layers

        word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
        # Pick device; allow explicit override
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # transformer.device = torch.device("cpu")
        # models.device = torch.device("cpu")
        cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
        print(device)

        # Load word map (word2id)
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        vocab_size = len(word_map)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

        # Load model checkpoint
        # Priority: explicit --checkpoint; otherwise fall back to common filenames under --path
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
        else:
            candidates = [
                'best_model.pth',
                'best_model.pth.tar',
                'BEST_checkpoint_HCNet_UCM.pth.tar',
                'model.pth',
                'checkpoint_HCNet_UCM.pth.tar'
            ]
            checkpoint_path = None
            for name in candidates:
                candidate_path = os.path.join(args.path, name)
                if os.path.exists(candidate_path):
                    checkpoint_path = candidate_path
                    break
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint found. Pass --checkpoint or place one of {} under {}".format(
                    candidates, args.path))

        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        print("Loading checkpoint:", checkpoint_path)

        map_location = "cpu" if args.load_on_cpu else str(device)
        checkpoint = load_checkpoint_with_fallback(
            checkpoint_path,
            map_location=map_location,
            cpu_fallback=args.load_cpu_fallback
        )

        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            decoder = checkpoint['decoder']
            decoder = decoder.to(device)
            decoder.eval()
            encoder = checkpoint['encoder']
            encoder = encoder.to(device)
            encoder.eval()
        else:
            attr_dir = os.path.join(args.data_folder, "attr")
            final_args = checkpoint.get('final_args', {})
            emb_dim = final_args.get('emb_dim', args.emb_dim)
            attention_dim = final_args.get('attention_dim', args.attention_dim)
            decoder_dim = final_args.get('decoder_dim', args.decoder_dim)
            n_heads = final_args.get('n_heads', args.n_heads)
            dropout = final_args.get('dropout', args.dropout)
            attr_topk = getattr(args, "attr_topk", None)
            if args.decoder_mode == "aoa":
                encoder = Encoder(NetType=args.encoder_mode)
                decoder = AoADecoder(
                    attention_dim=attention_dim,
                    embed_dim=emb_dim,
                    decoder_dim=decoder_dim,
                    vocab_size=vocab_size,
                    encoder_dim=1024,
                    num_heads=n_heads,
                    dropout=dropout,
                )
            elif args.decoder_mode == "aoa_hfam_cfim":
                encoder, decoder = build_aoa_hfam_cfim_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            elif args.decoder_mode == "aoa_attr_gcn":
                encoder, decoder = build_aoa_attr_gcn_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                    attr_dir=attr_dir,
                    attr_topk=attr_topk,
                )
            elif args.decoder_mode == "aoa_cfim_attr_gcn":
                encoder, decoder = build_aoa_cfim_attr_gcn_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                    attr_dir=attr_dir,
                    attr_topk=attr_topk,
                )
            elif args.decoder_mode == "aoa_hfam_attr_gcn":
                encoder, decoder = build_aoa_hfam_attr_gcn_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                    attr_dir=attr_dir,
                    attr_topk=attr_topk,
                )
            elif args.decoder_mode == "aoa_hfam_cfim_attr_gcn":
                encoder, decoder = build_aoa_hfam_cfim_attr_gcn_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                    attr_dir=attr_dir,
                    attr_topk=attr_topk,
                )
            elif args.decoder_mode == "aoa_mad_sap":
                encoder, decoder = build_aoa_mad_sap_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                    attr_dir=attr_dir,
                    attr_topk=attr_topk,
                    word_map=word_map,
                )
            elif args.decoder_mode == "aoa_hfam_cfim_mad_sap":
                encoder, decoder = build_aoa_hfam_cfim_mad_sap_models(
                    vocab_size=vocab_size,
                    embed_dim=emb_dim,
                    attention_dim=attention_dim,
                    decoder_dim=decoder_dim,
                    encoder_backbone=args.encoder_mode,
                    n_heads=n_heads,
                    dropout=dropout,
                    attr_dir=attr_dir,
                    attr_topk=attr_topk,
                    word_map=word_map,
                )
            else:
                raise ValueError(f"Unsupported decoder_mode: {args.decoder_mode}")

            encoder_state = checkpoint.get('encoder_state_dict')
            decoder_state = checkpoint.get('decoder_state_dict')
            if encoder_state is None or decoder_state is None:
                raise KeyError("Checkpoint missing encoder_state_dict/decoder_state_dict.")
            encoder.load_state_dict(encoder_state)
            decoder.load_state_dict(decoder_state)
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            encoder.eval()
            decoder.eval()

            # Normalization transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if args.decoder_mode in (
            "lstm_attention",
            "transformer_decoder",
            "aoa",
            "aoa_attr",
            "aoa_attr_fusion",
            "aoa_cfim",
            "aoa_attr_v3",
            "aoa_attr_gcn",
            "aoa_cfim_attr_gcn",
            "aoa_hfam_attr_gcn",
            "aoa_hfam_cfim_attr_gcn",
            "aoa_mad_sap",
            "aoa_hfam_cfim_mad_sap",
        ):
            metrics = evaluate_transformer(args, rev_word_map)

        print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
                  (args.decoder_mode, args.beam_size, metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"],
                   metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

        # Save metrics to txt
        # default output dir and name: result/<model_folder>_metrics.txt (model_folder = checkpoint所在目录名)
        out_dir = os.path.dirname(args.metrics_out) if args.metrics_out else "result"
        os.makedirs(out_dir, exist_ok=True)
        model_folder = os.path.basename(os.path.dirname(checkpoint_path)).strip(os.sep) or "model"
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        default_name = f"{model_folder}_metrics.txt" if model_folder else f"{base_name}_metrics.txt"
        out_path = args.metrics_out if args.metrics_out else os.path.join(out_dir, default_name)
        with open(out_path, "w") as f:
            f.write("decoder_mode: {}\n".format(args.decoder_mode))
            f.write("beam_size: {}\n".format(args.beam_size))
            for k in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]:
                if k in metrics:
                    f.write("{}: {:.4f}\n".format(k, metrics[k]))

        print("Metrics saved to:", out_path)

        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

        print("\n")
        print("\n")
        print("\n")
