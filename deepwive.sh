#!/usr/bin/env bash
# cuda:0
#source /liangyongsheng/anaconda3/bin/activate base && cd /liangyongsheng/Personal_File/BohuaiXiao/key_eval && sh deepwive.sh

python -u main.py \
    --trainer 'deepwive' \
    --config_file 'configs/deepwive.yaml' \
    --device 'cuda:0' \
    --resume