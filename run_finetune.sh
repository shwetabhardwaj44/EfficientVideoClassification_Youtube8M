
# FINETUNING STUDENT

#epc1
time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/code/train_finetune.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_finetune/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 6    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_finetune_after_1epc

mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc1
mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc1/train
mv ./model_HLSTM_TeaStud_every10_finetune/events* ./model_HLSTM_TeaStud_every10_finetune/backup_epc1/train/
cp -r ./model_HLSTM_TeaStud_every10_finetune/model.ckpt* ./model_HLSTM_TeaStud_every10_finetune/backup_epc1/
cp ./model_HLSTM_TeaStud_every10_finetune/check* ./model_HLSTM_TeaStud_every10_finetune/backup_epc1/

# #epc2
# time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/code/train_finetune.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_finetune/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 6    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_finetune_after_2epc

# mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc2
# mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc2/train
# mv ./model_HLSTM_TeaStud_every10_finetune/events* ./model_HLSTM_TeaStud_every10_finetune/backup_epc2/train/
# cp -r ./model_HLSTM_TeaStud_every10_finetune/model.ckpt* ./model_HLSTM_TeaStud_every10_finetune/backup_epc2/
# cp ./model_HLSTM_TeaStud_every10_finetune/check* ./model_HLSTM_TeaStud_every10_finetune/backup_epc2/

# #epc3
# time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/code/train_finetune.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_finetune/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 6    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_finetune_after_3epc

# mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc3
# mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/train
# mv ./model_HLSTM_TeaStud_every10_finetune/events* ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/train/
# cp -r ./model_HLSTM_TeaStud_every10_finetune/model.ckpt* ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/
# cp ./model_HLSTM_TeaStud_every10_finetune/check* ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/




