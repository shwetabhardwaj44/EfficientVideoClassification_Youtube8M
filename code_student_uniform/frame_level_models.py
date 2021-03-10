
"""Contains a collection of models which operate on variable-length sequences.
"""

import math
import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow.python.ops.rnn_cell_impl import RNNCell as RNNCell

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool("ppfs_normalize", False,
                  "Adds feature normalization to the PoorPoorFeatureSummationModel.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("input_features", 1024, "Input features size")
flags.DEFINE_integer("lstm_layers", 1, "Number of LSTM layers.")
flags.DEFINE_string("a_rate", 2,
                    "Rate of the atrous 2d convolutions")
flags.DEFINE_integer("num_conv2d_layers", 4, "Number of atrous 2-D layers")
flags.DEFINE_integer("filter_size", 10, "Size of atrous conv2d filter.")
flags.DEFINE_integer("max_num_frames", 300, "maximum number of frames in a video")
flags.DEFINE_integer("num_inputs_to_lstm", 20, "Number of final temporal inputs"
                     "to present to LSTM layer. Output of atrous operations.")
flags.DEFINE_integer("att_hid_size", 100, "att_hid_size")
xavier = tf.contrib.layers.xavier_initializer(uniform=False)

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.Variable(tf.random_normal(
        [feature_size, cluster_size],
        stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.Variable(
          tf.random_normal(
              [cluster_size], stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.Variable(tf.random_normal(
        [cluster_size, hidden1_size],
        stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.Variable(
          tf.random_normal(
              [hidden1_size], stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)


class HierarchicalLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames,  **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    print("Inside H-LSTM Model: create_model")
    print(model_input.shape)
    print(FLAGS.max_num_frames)
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    L2_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm

    num_frames_L1 = [tf.minimum(len_lower_lstm , tf.maximum(0, num_frames - int(len_lower_lstm) * i) ) for i in range(FLAGS.num_inputs_to_lstm)]

    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)

    with tf.variable_scope("classifier"):
      aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
      final_state_predictions = aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

    return state, final_state_predictions

  def create_model_inference(self, model_input, vocab_size, every_n, num_inputs_L1, num_frames,  **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    print("Inside H-LSTM Model: create_model_inference")
    print(model_input.shape)
    max_num_frames_student = FLAGS.max_num_frames/every_n
    print(max_num_frames_student)
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    L2_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    split_model_input = tf.split(model_input, num_inputs_L1, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = max_num_frames_student/num_inputs_L1
    zero = tf.cast(0, tf.int64)
    num_frames_L1 = [tf.minimum(tf.cast(len_lower_lstm, tf.int64) , tf.maximum(zero, num_frames - int(len_lower_lstm) * i) ) for i in range(num_inputs_L1)]
    #num_frames_L1 = [tf.minimum(len_lower_lstm , tf.maximum(0, num_frames - int(len_lower_lstm) * i) ) for i in range(num_inputs_L1)]

    L1_outputs = []
    for i in range(num_inputs_L1):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)

    with tf.variable_scope("classifier"):
      aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
      final_state_predictions = aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

    return state, final_state_predictions


class NetVLADModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames,  **unused_params):
    return

  def create_model_inference(self, model_input, vocab_size, every_n, num_frames,  **unused_params):
    return

class NeXtVLADModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames,  **unused_params):
    return

  def create_model_inference(self, model_input, vocab_size, every_n, num_frames,  **unused_params):
    return
