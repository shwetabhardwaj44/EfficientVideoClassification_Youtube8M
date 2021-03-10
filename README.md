# EfficentVideoClassification_Youtube8M
This repository contains code for the following paper:
> Shweta Bhardwaj, Mukundhan Srinivasan, Mitesh M. Khapra. *Efficient Video Classification Using Fewer Frames*. IEEE Conference on Computer Vision and Pattern Recognition 2019 [[https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf)].

- [Code organization](#code-organization)
- [Command examples](#command-examples)


# Code organization
- `run_train.sh:` Bash script to train Teacher and Student models together, generate logs in `output_HLSTM_TeaStud_every10_after_1epc` and model in `model_HLSTM_TeaStud_every10_train`. Sample command: 
* time CUDA_VISIBLE_DEVICES="0"  python -u code_student_uniform/code/train.py --train_data_pattern "./yt8m/train*.tfrecord" --train_dir  ./model_HLSTM_TeaStud_every10_train/   --frame_features True     --feature_names  "rgb, audio" --feature_sizes "1024, 128"   --model "HierarchicalLstmModel" --gpu 0    --batch_size 256  --num_inputs_to_lstm 20 --lstm_layers 2 --start_new_model True  --num_epochs 1  --every_n 10  &> output_HLSTM_TeaStud_every10_after_1epc


# Command Examples
```
bash run_finetune.sh
```
and here are some sample outputs from my local run:
```
Number of parameters: 369498
000: Acc-tr:  58.49, Acc-val: 56.92, L-tr: 1.1574, L-val: 1.1857
001: Acc-tr:  69.64, Acc-val: 68.24, L-tr: 0.8488, L-val: 0.8980
002: Acc-tr:  76.29, Acc-val: 73.24, L-tr: 0.6726, L-val: 0.7650
003: Acc-tr:  73.34, Acc-val: 70.56, L-tr: 0.7899, L-val: 0.9145
004: Acc-tr:  82.07, Acc-val: 77.42, L-tr: 0.5031, L-val: 0.6565
005: Acc-tr:  84.26, Acc-val: 79.33, L-tr: 0.4427, L-val: 0.6233
...
149: Acc-tr:  94.11, Acc-val: 80.31, L-tr: 0.1755, L-val: 0.9136
150: Acc-tr:  99.97, Acc-val: 85.87, L-tr: 0.0061, L-val: 0.5876
151: Acc-tr: 100.00, Acc-val: 86.31, L-tr: 0.0034, L-val: 0.5824
152: Acc-tr: 100.00, Acc-val: 86.25, L-tr: 0.0025, L-val: 0.5874
...
166: Acc-tr: 100.00, Acc-val: 86.44, L-tr: 0.0007, L-val: 0.6017
167: Acc-tr: 100.00, Acc-val: 86.52, L-tr: 0.0006, L-val: 0.6050
...
298: Acc-tr: 100.00, Acc-val: 86.43, L-tr: 0.0005, L-val: 0.5649
299: Acc-tr: 100.00, Acc-val: 86.41, L-tr: 0.0004, L-val: 0.5636
```
