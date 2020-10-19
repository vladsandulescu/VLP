"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import json
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle
import sys
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from misc.data_parallel import DataParallelImbalance
from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader

from vlp.lang_utils import language_eval
from sklearn.metrics import roc_auc_score
import pandas as pd


def validate(epoch_num, eval_model_recover_path, args, logger):
    val_sets = list(args.eval_split.split('+'))
    if "test_unseen" not in val_sets:
        val_sets.append("test_unseen")
    for val_set in val_sets:
        validate_set(epoch_num, eval_model_recover_path, args, logger, validation_set=val_set)


# epoch end validation and creating test predictions
def validate_set(epoch_num, eval_model_recover_path, args, logger, validation_set='dev'):
    if args.enable_butd:
        assert(args.len_vis_input == 100)
        args.region_bbox_file = os.path.join(args.image_root, args.region_bbox_file)
        args.region_det_file_prefix = os.path.join(args.image_root, args.region_det_file_prefix) if args.dataset in ('cc', 'coco', 'hm') and args.region_det_file_prefix != '' else ''

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    args.max_seq_length = args.max_len_b + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    tokenizer.max_len = args.max_seq_length

    bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(0, 0,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        new_segment_ids=args.new_segment_ids, truncate_config={
        'max_len_b': args.max_len_b, 'trunc_seg': 'b', 'always_truncate_tail': True},
        mode="bi", len_vis_input=args.len_vis_input, enable_butd=args.enable_butd,
        region_bbox_file=args.region_bbox_file, region_det_file_prefix=args.region_det_file_prefix,
        load_vqa_ann=False, load_hm_ann=True, test_mode=(validation_set.__contains__('test')))]

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    logger.info('Attempting to recover models from: {}'.format(eval_model_recover_path))
    if 0 == len(glob.glob(args.model_recover_path.strip())):
        logger.error('There are no models to recover. The program will exit.')
        sys.exit(1)
    for model_recover_path in glob.glob(eval_model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, task_idx=0,
            max_position_embeddings=512, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(-1),
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, tasks='hm')
        del model_recover

        if args.fp16:
            model.half()
            # cnn.half()
        model.to(device)
        # cnn.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            # cnn = torch.nn.DataParallel(cnn)

        torch.cuda.empty_cache()
        model.eval()
        # cnn.eval()

        eval_lst = seq2seq_loader.Img2txtDataset(
            args.src_file, args.image_root, validation_set, args.train_batch_size,
            tokenizer, args.max_seq_length, file_valid_jpgs=args.file_valid_jpgs,
            bi_uni_pipeline=bi_uni_pipeline, use_num_imgs=args.use_num_imgs,
            s2s_prob=args.s2s_prob, bi_prob=args.bi_prob,
            enable_butd=args.enable_butd, tasks=args.tasks).ex_list
        input_lines = eval_lst

        next_i = 0
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.eval_batch_size)
        predictions = []

        print('Epoch {}: start the HM evaluation...'.format(epoch_num))
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.eval_batch_size]
                if validation_set.__contains__('dev'):
                    buf = [(x[0], x[1], x[2]) for x in _chunk]
                    buf_id = [(x[0], x[2]) for x in _chunk]
                elif validation_set.__contains__('test'):
                    buf = [(x[0], x[1]) for x in _chunk]
                    buf_id = [(x[0]) for x in _chunk]
                else:
                    raise ValueError('Incorrect evaluation split.')

                next_i += args.eval_batch_size
                instances = []
                for instance in buf:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(
                        instances)
                    batch = [t.to(device) for t in batch]
                    if not validation_set.__contains__('test'):
                        input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, \
                        is_next, task_idx, img, vis_masked_pos, vis_pe, _ = batch
                    else:
                        input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, \
                        is_next, task_idx, img, vis_masked_pos, vis_pe = batch

                    if args.fp16:
                        img = img.half()
                        vis_pe = vis_pe.half()

                    if args.enable_butd:
                        conv_feats = img.data # Bx100x2048
                        vis_pe = vis_pe.data
                    else:
                        conv_feats, _ = cnn(img.data) # Bx2048x7x7
                        conv_feats = conv_feats.view(conv_feats.size(0), conv_feats.size(1),
                            -1).permute(0,2,1).contiguous()

                    proba, label = model(conv_feats, vis_pe, input_ids, segment_ids,
                        input_mask, lm_label_ids, None, is_next, masked_pos=masked_pos,
                        masked_weights=masked_weights, task_idx=task_idx,
                        vis_masked_pos=vis_masked_pos, drop_worst_ratio=0,
                        hm_inference=True)

                    proba = proba.detach().cpu().numpy()
                    label = label.detach().cpu().numpy()
                    if validation_set.__contains__('dev'):
                        for ind, (img, target) in enumerate(buf_id):
                            img_id = re.sub(r'^.*?img/', '', img).replace('.png', '')
                            predictions.append({'id': img_id, 'proba': proba[ind], 'label': label[ind],
                                                'target': target['label']})
                    elif validation_set.__contains__('test'):
                        for ind, img in enumerate(buf_id):
                            img_id = re.sub(r'^.*?img/', '', img).replace('.png', '')
                            predictions.append({'id': img_id, 'proba': proba[ind], 'label': label[ind]})
                    else:
                        raise ValueError('Incorrect evaluation split.')

                pbar.update(1)

        results = pd.DataFrame(predictions)
        if validation_set.__contains__('dev'):
            acc = (results.target == results.label).sum()/len(results)
            auroc = np.round(roc_auc_score(results.target.values, results.proba.values), 4)
            logger.info('Epoch {}: Val Acc = {}, Val AUROC = {}'.format(epoch_num, acc, auroc))
            print("Epoch {}: Val Acc = {}, Val AUROC = {}".format(epoch_num, acc, auroc))

        results_file = os.path.join(args.output_dir, 'hm-results-'+model_recover_path.split('/')[-2])+'-'+validation_set+'-'+model_recover_path.split('/')[-1].split('.')[-2]+'.csv'
        results.proba = results.proba.round(4)
        results.to_csv(results_file, header=True, index=False)
