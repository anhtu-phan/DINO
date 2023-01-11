#!/bin/bash

NAME="dino"
PROJECT_NAME="visdrone"
coco_path = ./dataset/$PROJECT_NAME

cd models/dino/ops
python setup.py build install
cd ../../..

python -m torch.distributed.launch main.py --output_dir logs/$PROJECT_NAME/$NAME -c config/DINO/DINO_4scale.py --coco_path $coco_path --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --pretrain_model_path checkpoint/checkpoint0033_4scale.pth --finetune_ignore label_enc.weight class_embed --wandb_project_name $PROJECT_NAME --wandb_name $NAME