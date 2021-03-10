# EfficientVideoClassification_Youtube8M
This repository contains code for the following paper:
> Shweta Bhardwaj, Mukundhan Srinivasan, Mitesh M. Khapra. *Efficient Video Classification Using Fewer Frames*. IEEE Conference on Computer Vision and Pattern Recognition 2019 [https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf)].

- [Requirements](#Requirements)
- [Dataset](#Dataset)
- [Code organization](#code-organization)
- [Command Examples](#command-examples)
- [Upcoming](#Upcoming)

# Requirements
* Tensorflow: 1.3.0
* Python: 2.7.18 (supports Python3)
* cPickle: 1.71

# Dataset
- Youtube-8M Large-Scale Video Understanding: http://research.google.com/youtube8m/download.html
- YouTube-8M small sample link: (https://drive.google.com/drive/folders/1XY63FfJY7k7YD3WrQ8DN7L6kTiX4BxlK?usp=sharing). Download dataset and save in `yt8m` folder.

- No. of classes (in current code): 4716
- 2017 Version: 7.0M videos, 4716 classes, 3.4 labels/video, 3.2B audio-visual features
- Note: This code supports all versions of dataset. In ```code_student_uniform/readers.py```, You can change ```video_id``` to ```id``` and set ```num_classes``` as per the dataset version.

# Code Organization
Bash Scripts for end-to-end training:
- `run_train.sh`: Bash script to train Teacher and Student network Parallely, generate logs in `output_HLSTM_TeaStud_every10_after_Nepc` and model in `model_HLSTM_TeaStud_every10_train`
- `run_validate.sh`: Bash script to evaluate only student network saved in `model_HLSTM_TeaStud_every10_train` on validation set and generate logs in `validate_HLSTM_TeaStud_every10_train_after_Nepc`
- `run_convert_model.sh`: Bash script for converting stored Teacher-Student meta-graph in `model_HLSTM_TeaStud_every10_train`, to Student meta-graph in  `model_HLSTM_TeaStud_every10_finetune`
- `run_finetune.sh`: Bash script for fine-tuning pre-trained Student in `model_HLSTM_TeaStud_every10_finetune` and generate logs in `output_HLSTM_TeaStud_every10_finetune_after_Nepc`
- `run_evaluate.sh`: Bash script for evaluating fine-tuned Student and generate logs in `eval_HLSTM_TeaStud_every10_finetune_after_Nepc`

Main Code Files:
- `code_student_uniform/train.py`: Binary for Parallel training of Teacher and Student Tensorflow models (Hierarchical LSTMs) on YouTube-8M dataset.
- `code_student_uniform/train_convert_model.py`: Binary for converting Meta-graph from Teacher-Student to Student in the Network on YouTube-8M dataset.
- `code_student_uniform/train_finetune.py`: Binary for training(fine-tuning) pre-trained Student Tensorflow models on YouTube-8M dataset.
- `code_student_uniform/frame_level_models.py`: Contains a collection of Models (with Teacher and Student architectures) which operate on variable-length sequences.
- `code_student_uniform/validate.py`: Binary for evaluating Student Tensorflow models in the Teacher-Student architecture on the YouTube-8M dataset.
- `code_student_uniform/eval_finetune.py`: Binary for evaluating Student Tensorflow models after fine-tuning on the YouTube-8M dataset.

# Command Examples
```
bash run_train.sh
```
and here are some sample outputs on a smaller subset of training data, from my local run (macOS 10.14.6):
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
INFO:tensorflow:Restoring parameters from ./model_HLSTM_TeaStud_every10_train/model.ckpt-0
INFO:tensorflow:Starting standard services.
INFO:tensorflow:Saving checkpoint to path ./model_HLSTM_TeaStud_every10_train/model.ckpt
INFO:tensorflow:Starting queue runners.
INFO:tensorflow:/job:master/task:0: Entering training loop.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:global_step/sec: 0.00674478
INFO:tensorflow:/job:master/task:0: training step 2| Hit@1: 0.00| PERR: 0.00| GAP: 0.00| Teacher_Loss: 1914.1583| L_REP: 1.1446424| L_PRED: 2.1074477e-05| L_CE: 1914.1437
INFO:tensorflow:/job:master/task:0: training step 4| Hit@1: 0.09| PERR: 0.04| GAP: 0.01| Teacher_Loss: 1910.0898| L_REP: 1.4490945| L_PRED: 2.739967e-05| L_CE: 1911.522

```

# Upcoming:
Support for Cluster Based Methods : NetVLAD, NeXtVLAD
