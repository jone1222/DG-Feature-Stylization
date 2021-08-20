#!/bin/bash

dataset_root="./downloaded_datasets/Domain_Generalization"
output_path="./output/featstyle_pacs_pcs2a"
model_path="./resnet18_bestmodels/pcs2a"
seed=0

# Sample test script for pcs2a ( Source : Photo, Cartoon, Sketch / Target : Art_Painting )
python ./tools/train.py --root ${dataset_root} \
 --trainer FeatureStylization \
 --source-domains photo cartoon sketch \
 --target-domains art_painting \
 --dataset-config-file configs/datasets/dg/pacs.yaml \
 --config-file configs/trainers/dg/featstyle/pacs.yaml \
 --output-dir ${output_path} \
 --seed ${seed} \
 --eval-only \
 --model-dir ${model_path} \
 MODEL.BACKBONE.NAME resnet18_stylize