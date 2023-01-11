#!/bin/bash

NAME="dino-pretrain"
PROJECT_NAME="protector"
coco_path=/usr/src/datasets/$PROJECT_NAME

python -m torch.distributed.launch main.py --output_dir logs/$PROJECT_NAME/$NAME -c config/DINO/DINO_4scale.py --coco_path $coco_path --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --pretrain_model_path checkpoint/visdrone_4scale.pth --finetune_ignore label_enc.weight class_embed --wandb_project_name $PROJECT_NAME --wandb_name $NAME