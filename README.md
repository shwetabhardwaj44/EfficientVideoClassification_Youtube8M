# EfficentVideoClassification_Youtube8M
This repository contains code for the following paper:
> Shweta Bhardwaj, Mukundhan Srinivasan, Mitesh M. Khapra. *Efficient Video Classification Using Fewer Frames*. IEEE Conference on Computer Vision and Pattern Recognition 2019 [[https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf)].

- [Requirements](#Requirements)
- [Dataset](#Dataset)
- [Code organization](#code-organization)
- [Command Examples](#command-examples)
- [Notes](#Notes)
- [Upcoming](#Upcoming)
- 
# Requirements
* Tensorflow: 1.3.0
* Python: 2.7.18 (supports Python3 also)
* cPickle: 1.71

# Dataset
YouTube-8M Data: http://research.google.com/youtube8m/download.html
YouTube-8M small sample link: (https://drive.google.com/drive/folders/1XY63FfJY7k7YD3WrQ8DN7L6kTiX4BxlK?usp=sharing)

No. of classes (in current code): 4716
2017 Version: 7.0M videos, 4716 classes, 3.4 labels/video, 3.2B audio-visual features
Note: This code supports all versions of dataset. 
In ```code_student_uniform/code/readers.py```, You can change ```id``` to ```video_id``` and ```num_classes``` to required.

# Code Organization
- `run_train.sh`: Bash script to train Teacher and Student network together, generate logs in `output_HLSTM_TeaStud_every10_after_Nepc` and model in `model_HLSTM_TeaStud_every10_train`
- `run_validate.sh`: Bash script to evaluate only student network saved in `model_HLSTM_TeaStud_every10_train` on validation set and generate logs in `validate_HLSTM_TeaStud_every10_train_after_Nepc`
- `run_convert_model.sh`:
- `run_finetune.sh`:
- `run_evaluate.sh`:

Main Code Files:
- `code_student_uniform/code/train.py`: Binary for training dynamic Teacher and Student Tensorflow models on YouTube-8M dataset.
- `code_student_uniform/code/train_convert_model.py`: Binary for converting Meta-graph from Teacher-Student to Student in the Network on YouTube-8M dataset.
- `code_student_uniform/code/train_finetune.py`: Binary for training(fine-tuning) pre-trained Student Tensorflow models on YouTube-8M dataset.
- `code_student_uniform/code/frame_level_models.py`: Contains a collection of Models (with Teacher and Student architectures) which operate on variable-length sequences.
- `code_student_uniform/code/validate.py`: Binary for evaluating Student Tensorflow models in the Teacher-Student architecture on the YouTube-8M dataset.
- `code_student_uniform/code/eval_finetune.py`: Binary for evaluating Student Tensorflow models after fine-tuning on the YouTube-8M dataset.

# Command Examples
```
bash run_train.sh
```
and here are some sample outputs on a smaller subset of training data (data), from my local run:
```
Key: video_level_classifier_model Value: MoeModel
Key: train_dir Value: ./model_HLSTM_TeaStud_every10_train/
Key: input_features Value: 1024
Key: dbof_pooling_method Value: max
Key: start_new_model Value: True
Key: learning_rate_decay_examples Value: 4000000
Key: base_learning_rate Value: 0.001
Key: moe_num_mixtures Value: 2
Key: filter_size Value: 10
Key: iterations Value: 30
Key: num_epochs Value: 1
Key: lstm_layers Value: 2
Key: feature_sizes Value: 1024, 128
Key: num_inputs_to_lstm Value: 20
Key: bagging Value: False
Key: a_rate Value: 2
Key: max_num_frames Value: 300
Key: every_n Value: 10
Key: dbof_add_batch_norm Value: True
Key: feature_names Value: rgb, audio
Key: gpu Value: 0
Key: lstm_cells Value: 1024
Key: log_device_placement Value: False
Key: clip_gradient_norm Value: 1.0
Key: sample_random_frames Value: True
Key: optimizer Value: AdamOptimizer
Key: frame_features Value: True
Key: regularization_penalty Value: 2
Key: dropout Value: 0.5
Key: batch_size Value: 256
Key: att_hid_size Value: 100
Key: num_hidden_units Value: 1024
Key: learning_rate_decay Value: 1
Key: label_loss Value: CrossEntropyLoss
Key: train_data_pattern Value: ./yt8m/train*.tfrecord
Key: ppfs_normalize Value: False
Key: model Value: HierarchicalLstmModel
Key: num_readers Value: 4
Key: dbof_hidden_size Value: 1024
Key: num_conv2d_layers Value: 4
Key: dbof_cluster_size Value: 8192
INFO:tensorflow:/job:master/task:0: Tensorflow version: 1.3.0.
INFO:tensorflow:/job:master/task:0: Flag 'start_new_model' is set. Building a new model.
INFO:tensorflow:Using batch size of 256 for training.
INFO:tensorflow:Number of training files: 5.
==================================
Inside H-LSTM Model: create_model
(?, 300, 1152)
300
Confirming Shapes of batch_size, predictions and labels_batch:
(256, TensorShape([Dimension(None), Dimension(4716)]), TensorShape([Dimension(None), Dimension(4716)]))
Trainable Parameters of Teacher:
[u'model/RNN_L1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0', u'model/RNN_L1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0', u'model/RNN_L1/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0', u'model/RNN_L1/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0', u'model/RNN_L2/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0', u'model/RNN_L2/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0', u'model/RNN_L2/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0', u'model/RNN_L2/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0', u'model/classifier/gates/weights:0', u'model/classifier/experts/weights:0', u'model/classifier/experts/biases:0']
Inside H-LSTM Model: create_model_inference
(?, 30, 1152)
30
Confirming Shapes of batch_size, predictions and labels_batch:
(256, TensorShape([Dimension(None), Dimension(4716)]), TensorShape([Dimension(None), Dimension(4716)]))
Trainable Parameters of Student:
[u'model_student/RNN_L1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0', u'model_student/RNN_L1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0', u'model_student/RNN_L1/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0', u'model_student/RNN_L1/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0', u'model_student/RNN_L2/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0', u'model_student/RNN_L2/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0', u'model_student/RNN_L2/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0', u'model_student/RNN_L2/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0', u'model_student/classifier/gates/weights:0', u'model_student/classifier/experts/weights:0', u'model_student/classifier/experts/biases:0']
INFO:tensorflow:/job:master/task:0: Built graph.
INFO:tensorflow:/job:master/task:0: Starting managed session.
INFO:tensorflow:Starting standard services.
INFO:tensorflow:Saving checkpoint to path ./model_HLSTM_TeaStud_every10_train/model.ckpt
INFO:tensorflow:Starting queue runners.
INFO:tensorflow:/job:master/task:0: Entering training loop.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:global_step/sec: 0.00674478
Vars to train
INFO:tensorflow:/job:master/task:0: training step 2| Hit@1: 0.00| PERR: 0.00| GAP: 0.00| Loss: 1914.3633| student_loss 1915.5684
INFO:tensorflow:global_step/sec: 0.0109319
INFO:tensorflow:Recording summary at step 2.
INFO:tensorflow:Recording summary at step 3.
Vars to train
INFO:tensorflow:/job:master/task:0: training step 4| Hit@1: 0.07| PERR: 0.04| GAP: 0.01| Loss: 1911.4663| student_loss 1913.5865
INFO:tensorflow:global_step/sec: 0.0144734
INFO:tensorflow:Recording summary at step 5.
INFO:tensorflow:global_step/sec: 0.00882497
Vars to train
INFO:tensorflow:/job:master/task:0: training step 6| Hit@1: 0.20| PERR: 0.16| GAP: 0.02| Loss: 590.33594| student_loss 1921.3817
INFO:tensorflow:Recording summary at step 6.
INFO:tensorflow:global_step/sec: 0.00921367
INFO:tensorflow:Recording summary at step 7.
Vars to train
INFO:tensorflow:/job:master/task:0: training step 8| Hit@1: 0.22| PERR: 0.13| GAP: 0.03| Loss: 32.490585| student_loss 13173.261
INFO:tensorflow:global_step/sec: 0.0164792
INFO:tensorflow:Recording summary at step 9.
```

# Upcoming:
Support for Cluster Based Methods : NetVLAD, NeXtVLAD

