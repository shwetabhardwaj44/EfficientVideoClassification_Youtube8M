#!/bin/bash
mkdir ./model_HLSTM_TeaStud_every10_train

# Training of Dynamic Teacher and Student on Train Data:

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/train.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 0    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model True  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_after_1epc

# Saving checkpoint after every epoch of training!
mkdir ./model_HLSTM_TeaStud_every10_train/backup_epc1
cp -r ./model_HLSTM_TeaStud_every10_train/model.ckpt* ./model_HLSTM_TeaStud_every10_train/backup_epc1/
mv ./model_HLSTM_TeaStud_every10_train/events* ./model_HLSTM_TeaStud_every10_train/backup_epc1/
cp ./model_HLSTM_TeaStud_every10_train/check* ./model_HLSTM_TeaStud_every10_train/backup_epc1/

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/train.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 0    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_after_2epc

mkdir ./model_HLSTM_TeaStud_every10_train/backup_epc2
cp -r ./model_HLSTM_TeaStud_every10_train/model.ckpt* ./model_HLSTM_TeaStud_every10_train/backup_epc2/
mv ./model_HLSTM_TeaStud_every10_train/events* ./model_HLSTM_TeaStud_every10_train/backup_epc2/
cp ./model_HLSTM_TeaStud_every10_train/check* ./model_HLSTM_TeaStud_every10_train/backup_epc2/

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/train.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 0    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_after_3epc

mkdir ./model_HLSTM_TeaStud_every10_train/backup_epc3
cp -r ./model_HLSTM_TeaStud_every10_train/model.ckpt* ./model_HLSTM_TeaStud_every10_train/backup_epc3/
mv ./model_HLSTM_TeaStud_every10_train/events* ./model_HLSTM_TeaStud_every10_train/backup_epc3/
cp ./model_HLSTM_TeaStud_every10_train/check* ./model_HLSTM_TeaStud_every10_train/backup_epc3/

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/train.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 0    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_after_4epc

mkdir ./model_HLSTM_TeaStud_every10_train/backup_epc4
cp -r ./model_HLSTM_TeaStud_every10_train/model.ckpt* ./model_HLSTM_TeaStud_every10_train/backup_epc4/
mv ./model_HLSTM_TeaStud_every10_train/events* ./model_HLSTM_TeaStud_every10_train/backup_epc4/
cp ./model_HLSTM_TeaStud_every10_train/check* ./model_HLSTM_TeaStud_every10_train/backup_epc4/

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/train.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 0    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_after_5epc

mkdir ./model_HLSTM_TeaStud_every10_train/backup_epc5
cp -r ./model_HLSTM_TeaStud_every10_train/model.ckpt* ./model_HLSTM_TeaStud_every10_train/backup_epc5/
mv ./model_HLSTM_TeaStud_every10_train/events* ./model_HLSTM_TeaStud_every10_train/backup_epc5/
cp ./model_HLSTM_TeaStud_every10_train/check* ./model_HLSTM_TeaStud_every10_train/backup_epc5/
