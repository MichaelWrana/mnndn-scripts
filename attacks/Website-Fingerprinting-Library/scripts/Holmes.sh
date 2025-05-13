#!/bin/bash

set -e

cd ./website-attacks/Website-Fingerprinting-Library/

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

function pyenv_virtualenv_exists {
    pyenv virtualenvs --bare | grep -q "^holmes-env$"
}

if pyenv_virtualenv_exists; then
    echo "Step 1: Virtual environment 'holmes-env' already exists. Activating it..."
else
    echo "Step 1: Creating a pyenv virtual environment named 'holmes-env' with Python 3.8.0"
    pyenv virtualenv 3.8.0 holmes-env
fi


echo "Step 2: Activate the 'holmes-env' virtual environment"
pyenv activate holmes-env

# echo "Step 3: Install dependencies from requirements.txt"
# pip install -r requirements.txt


dataset=CW
attr_method=DeepLiftShap 

# for filename in train valid
# do 
#     python -u exp/data_analysis/temporal_extractor.py \
#       --dataset ${dataset} \
#       --seq_len 10000 \
#       --in_file ${filename}
# done

echo "################## Step 1: temporal_exractor.py ##################"
python -u ./exp/data_analysis/temporal_extractor.py \
      --dataset ${dataset} \
      --seq_len 10000 \
      --in_file train


echo "################## Step 2: train.py ##################"
python -u ./exp/train.py \
  --dataset ${dataset} \
  --model RF \
  --device cuda:0 \
  --train_file temporal_train \
  --feature TAM \
  --seq_len 1000 \
  --train_epochs 30 \
  --batch_size 200 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name temporal


echo "################## Step 3: feature_attr.py ##################"
python -u ./exp/data_analysis/feature_attr.py \
  --dataset ${dataset} \
  --model RF \
  --in_file temporal_train \
  --device cpu \
  --feature TAM \
  --seq_len 1000 \
  --save_name temporal \
  --attr_method ${attr_method}


# for filename in train valid
# do 
#     python -u exp/dataset_process/data_augmentation.py \
#       --dataset ${dataset} \
#       --model RF \
#       --in_file ${filename} \
#       --attr_method ${attr_method}
# done

echo "################## Step 4: data_augmentation.py ##################"
python -u ./exp/dataset_process/data_augmentation.py \
    --dataset ${dataset} \
    --model RF \
    --in_file train \
    --attr_method ${attr_method}


# for filename in aug_train aug_valid test
# do 
#     python -u exp/dataset_process/gen_taf.py \
#       --dataset ${dataset} \
#       --seq_len 10000 \
#       --in_file ${filename}
# done

echo "################## Step 5: gen_taf.py ##################"
for filename in aug_train test
do 
    python -u ./exp/dataset_process/gen_taf.py \
      --dataset ${dataset} \
      --seq_len 10000 \
      --in_file ${filename}
done


# python -u exp/train.py \
#   --dataset ${dataset} \
#   --model Holmes \
#   --device cuda:6 \
#   --train_file taf_aug_train \
#   --valid_file taf_aug_valid \
#   --feature TAF \
#   --seq_len 2000 \
#   --train_epochs 30 \
#   --batch_size 256 \
#   --learning_rate 5e-4 \
#   --loss SupConLoss \
#   --optimizer AdamW \
#   --eval_metrics Accuracy Precision Recall F1-score \
#   --save_metric F1-score \
#   --save_name max_f1

echo "################## Step 6: train.py part-2 ##################"
python -u ./exp/train.py \
  --dataset ${dataset} \
  --model Holmes \
  --device cuda:0 \
  --train_file taf_aug_train \
  --feature TAF \
  --seq_len 2000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --loss SupConLoss \
  --optimizer AdamW \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

echo "################## Step Run spatial analysis ##################"
python -u exp/data_analysis/spatial_analysis.py \
  --dataset ${dataset} \
  --model Holmes \
  --device cuda:0 \
  --valid_file taf_aug_train \
  --feature TAF \
  --seq_len 2000 \
  --batch_size 256 \
  --save_name max_f1

echo "################## Step 7: gen_taf.py + test.py ##################"
# for percent in {20..100..10}
# do
#     python -u ./exp/dataset_process/gen_taf.py \
#         --dataset ${dataset} \
#         --seq_len 10000 \
#         --in_file test
#         # --in_file test_p${percent}

#     # python -u exp/test.py \
#     # --dataset ${dataset} \
#     # --model Holmes \
#     # --device cuda:6 \
#     # --valid_file taf_aug_valid \
#     # --test_file taf_test_p${percent} \
#     # --feature TAF \
#     # --seq_len 2000 \
#     # --batch_size 256 \
#     # --eval_method Holmes \
#     # --eval_metrics Accuracy Precision Recall F1-score \
#     # --load_name max_f1 \
#     # --result_file test_p${percent}

#     python -u ./exp/test.py \
#     --dataset ${dataset} \
#     --model Holmes \
#     --device cuda:0 \
#     --test_file taf_test_p${percent} \
#     --feature TAF \
#     --seq_len 2000 \
#     --batch_size 256 \
#     --eval_method Holmes \
#     --eval_metrics Accuracy Precision Recall F1-score \
#     --load_name max_f1 \
#     --result_file test_p${percent}
# done

python -u ./exp/dataset_process/gen_taf.py \
  --dataset ${dataset} \
  --seq_len 10000 \
  --in_file test
        # --in_file test_p${percent}

    # python -u exp/test.py \
    # --dataset ${dataset} \
    # --model Holmes \
    # --device cuda:6 \
    # --valid_file taf_aug_valid \
    # --test_file taf_test_p${percent} \
    # --feature TAF \
    # --seq_len 2000 \
    # --batch_size 256 \
    # --eval_method Holmes \
    # --eval_metrics Accuracy Precision Recall F1-score \
    # --load_name max_f1 \
    # --result_file test_p${percent}

python -u ./exp/test.py \
    --dataset ${dataset} \
    --model Holmes \
    --device cuda:0 \
    --test_file taf_test \
    --feature TAF \
    --seq_len 2000 \
    --batch_size 256 \
    --eval_method Holmes \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file test