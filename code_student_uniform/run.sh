#!/bin/bash

# Command line flags:
# $1 - Mode of operation. One of "video" and "frame"
# $2 - GPU number. One of 0 or 1
# $3 - Model number. A unique identifier number for each model.

# Specify Mode of features
if [ -z "$1" ]; then
  echo "Please enter the mode for features, e.g. ./run.sh video 0 1"
  echo "Feature Choices: video|frame"
  exit 0
else
  FEATURETYPE=$1
  shift
fi

DATA_SOURCE="./frame_data/train/*.tfrecord"

# Run different feature modes with different hyper-parameters

if [ "$FEATURETYPE" == "video" ]; then
  # Video level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=./data_video/train*.tfrecord --train_dir=./models/video_level_model_$2  --feature_names "mean_rgb" --frame_features False  --model "LogisticModel" --gpu $1 > output-video-$2-$4  
elif [ "$FEATURETYPE" == "HierarchicalLstm" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE \
  --train_dir=./models/frame_level_model_HierarchicalLstmModel_$2 --feature_names  "rgb" --frame_features True  \
  --model "HierarchicalLstmModel" --gpu $1 --batch_size 128 --lstm_layers $3 &> output-frame-HierarchicalLstmModel-$2-$4  
elif [ "$FEATURETYPE" == "DilatedConvolution" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_DilatedConvolution_$2 \
   --feature_names  "rgb" --frame_features True  --model "DilatedConvolutionModel" --gpu $1 \
   --batch_size 128 &> output-frame-DilatedConvolution-$2-$4
elif [ "$FEATURETYPE" == "Convolution" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_Convolution_$2 \
   --feature_names  "rgb" --frame_features True  --model "ConvolutionModel" --gpu $1 \
   --batch_size 128 --num_inputs_to_lstm 20 &> output-frame-Convolution-$2-$4  

elif [ "$FEATURETYPE" == "PoorNoneAttention" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_PoorNoneAttention_$2 \
   --feature_names  "rgb" --frame_features True  --model "PoorNoneAttentionModel" --gpu $1 \
   --batch_size 64 --num_inputs_to_lstm 20  --lstm_layers $3  &> output-frame-PoorNoneAttention-$2-$4

elif [ "$FEATURETYPE" == "PoorPoorFeatureAttention" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_PoorPoorAttention-$2 \
  --feature_names  "rgb" --frame_features True  --model "PoorPoorFeatureAttentionModel" --gpu $1 \
  --batch_size 64 --num_inputs_to_lstm 20 --lstm_layers $3 &> output-frame-PoorPoorAttention-$2-$4

elif [ "$FEATURETYPE" == "PoorRichAttention" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_PoorRichAttention-$2-$4 \
  --feature_names  "rgb" --frame_features True  --model "PoorRichAttentionModel" --gpu $1 \
  --batch_size 128 --num_inputs_to_lstm 20 --lstm_layers $3 &> output-frame-PoorRichAttention-$2-$4

elif [ "$FEATURETYPE" == "PoorPoorFeatureSummation" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_PoorPoorFeatureSummation-$2 \
  --feature_names  "rgb" --frame_features True  --model "PoorPoorFeatureSummationModel" --gpu $1 \
  --batch_size 128 --num_inputs_to_lstm 20 --lstm_layers $3 &> output-frame-PoorPoorSummation-$2


elif [ "$FEATURETYPE" == "Summary" ]; then
  # Frame level features
  echo $1
  echo $2
  time python -u ./ishu_code/train.py --train_data_pattern=$DATA_SOURCE \
  --train_dir=./models/frame_level_SummaryModel_$2 --feature_names "rgb" --frame_features True \
  --model "SummaryModel" --batch_size 256 --gpu $1 &> output-frame-Summary-$2-$4

elif [ "$FEATURETYPE" == "WindowDetection" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_WindowDetection_$2 \
  --feature_names  "rgb" --frame_features True  --model "WindowDetectionModel" --gpu $1 \
  --batch_size 128 --num_inputs_to_lstm 20 --lstm_layers $3 &> output-frame-WindowDetection-$2

elif [ "$FEATURETYPE" == "HierarchicalWindowDetection" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_HierarchicalWindowDetection_$2 \
  --feature_names  "rgb, audio" --feature_sizes="1024, 128" --frame_features True  --model "HierarchicalWindowDetectionModel" --gpu $1 \
  --batch_size 128 --num_inputs_to_lstm 20 --lstm_layers $3 &> output-frame-HierarchicalWindowDetection-$2

elif [ "$FEATURETYPE" == "HierarchicalResidual" ]; then
  # Frame level features
  echo $1
  echo $2
  time  python -u ./code/train.py --train_data_pattern=$DATA_SOURCE --train_dir=./models/frame_level_model_HierarchicalResidual_$2 \
  --feature_names  "rgb, audio" --feature_sizes="1024, 128" --frame_features True  --model "HierarchicalResidualModel" --gpu $1 \
  --batch_size 128 --num_inputs_to_lstm 20 --lstm_layers $3 &> output-frame-HierarchicalResidual-$2


else
  echo "Invalid options"
fi
