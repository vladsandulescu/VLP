#!/bin/sh

export PYTHONPATH=$PYTHONPATH:./

python ./vlp/run_img2txt_dist.py \
  --output_dir /work1/paupo/playground/hmm/lib/VLP/output/hm \
  --model_recover_path /work1/paupo/playground/hmm/lib/VLP/model/pretrained_model/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
  --image_root /work1/paupo/playground/hmm/data/img \
  --src_file /work1/paupo/playground/hmm/data/ \
  --region_det_file_prefix /work1/paupo/playground/hmm/data/imgfeat/region_feat_gvd_wo_bgd/feat_cls_1000/hm_detection_vg_100dets_vlp_checkpoint_trainval \
  --region_bbox_file /work1/paupo/playground/hmm/data/imgfeat/region_feat_gvd_wo_bgd/raw_bbox/hm_detection_vg_100dets_vlp_checkpoint_trainval_bbox.h5 \
  --dataset hm \
  --do_train \
  --split train+dev \
  --eval_split dev \
  --train_batch_size 64 \
  --eval_batch_size 64 \
  --tasks hm \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --new_segment_ids \
  --always_truncate_tail \
  --amp \
  --enable_butd \
  --s2s_prob 0 \
  --bi_prob 1 \
  --mask_prob 0 \
  --max_pred 1 \
  --num_workers 4
