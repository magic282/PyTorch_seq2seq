#!/bin/bash

set -x

DATAHOME=${@:(-2):1}
EXEHOME=${@:(-1):1}

SAVEPATH=$DATAHOME/models/s2s

cd $EXEHOME

python train.py -data na \
       -save_path $SAVEPATH \
       -log_home $SAVEPATH \
       -online_process_data \
       -train_src $DATAHOME/train/train.article.txt -src_vocab $DATAHOME/train/source.vocab \
       -train_tgt $DATAHOME/train/train.title.txt -tgt_vocab $DATAHOME/train/target.vocab \
       -dev_input_src $DATAHOME/dev/valid.article.filter.txt.8k -dev_ref $DATAHOME/dev/valid.title.filter.txt.8k \
       -layers 1 -enc_rnn_size 512 -brnn -word_vec_size 300 -dropout 0.5 \
       -batch_size 64 -beam_size 1 \
       -epochs 20 \
       -optim adam -learning_rate 0.001 \
       -gpus 0 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 15000 -eval_per_batch 1500 \
       -seed 12345 -cuda_seed 12345

#umount /mnt
