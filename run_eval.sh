
# Evaluate after epoch 3 of Fine-tuning

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/code/eval_finetune.py --eval_data_pattern "./yt8m/validate*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 1    --batch_size 512  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model False  --num_epochs 1  --every_n 10  --top_k 20 --run_once True &> eval_HLSTM_TeaStud_every10_finetune_after_3epc

mkdir ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/evals
mv ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/events* ./model_HLSTM_TeaStud_every10_finetune/backup_epc3/evals/

