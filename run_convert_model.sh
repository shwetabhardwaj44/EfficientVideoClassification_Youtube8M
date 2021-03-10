

time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/code/train_convert_model.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 6    --batch_size 128  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model True  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_convertModel
