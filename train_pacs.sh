#!/bin/bash

dataset_root="./downloaded_datasets/Domain_Generalization"
output_path="./output/featstyle_pacs_pcs2a"
seed=0

# Sample train script for pcs2a ( Source : Photo, Cartoon, Sketch / Target : Art_Painting )
python tools/train.py --root ${dataset_root} \
  --trainer FeatureStylization \
  --source-domains photo cartoon sketch \
  --target-domains art_painting \
  --dataset-config-file configs/datasets/dg/pacs.yaml \
  --config-file configs/trainers/dg/featstyle/pacs.yaml \
  --output-dir ${output_path} \
  --seed ${seed} \
  TEST.FINAL_MODEL best_val \
  DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
  MODEL.BACKBONE.NAME resnet18_stylize \
  MODEL.BACKBONE.PRETRAINED True \
  MODEL.BACKBONE.FREEZE_BN True \
  MODEL.CLASSIFIER.BIAS False \
  TRAINER.FEATSTYLE.LMDA_CONTRA 0.3 \
  TRAINER.FEATSTYLE.TAU_CONTRA 0.15 \
  TRAINER.FEATSTYLE.LMDA_CONSISTENCY 12 \
  TRAINER.FEATSTYLE.TAU_CONSISTENCY 0.5 \
  TRAINER.FEATSTYLE.MEASURE_CONSISTENCY kl \
  TRAINER.FEATSTYLE.SCALING_FACTOR 10 \
  TRAINER.FEATSTYLE.ENCODE_MODE pooling