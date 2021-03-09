
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
#from tensorflow.models.rnn import  rnn_cell as RNNCell



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

class SimpleSingle2048LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames,  batch_size, **unused_params):
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
    print("Inside H-LSTM model Teacher of SimpleSingle2048LstmModel")
    print(model_input.shape)    
    print(FLAGS.max_num_frames)

    lstm_size = FLAGS.lstm_cells
    number_of_layers = 1

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

    #zero = tf.cast(0, tf.int64)
    #num_frames_L1 = [tf.minimum(tf.cast(len_lower_lstm, tf.int64) , tf.maximum(zero, num_frames - int(len_lower_lstm) * i) ) for i in range(FLAGS.num_inputs_to_lstm)]

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
      outputs, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
    print("state and output in teaher")
    print(state.shape)
    print(outputs[-1].shape)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return state, aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)


  def create_model_inference(self, model_input, vocab_size, num_frames, batch_size, **unused_params):

    print("****************Inside Student Model of SImpleSingleLstmModel ************************")
    print(model_input.shape)
    print(batch_size)
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    ## Simple LSTM student
    simple_lstm = tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)

    loss = 0.0
    l2_penalty=1e-8
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(simple_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)
    #student_state = slim.fully_connected(state, 4096, activation_fn=None, weights_regularizer=slim.l2_regularizer(l2_penalty))
    student_state = state
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=student_state,
        vocab_size=vocab_size,
        **unused_params)


  def create_student(self, model_input, vocab_size, num_frames, batch_size, **unused_params):
    """Creates a smaller LSTM model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
    Final encoding of LSTM. The dimensions of the tensor are
      'batch_size' x 'lstm_size'.
    """
    print("**************** Inside Student Model of SImpleSingleLstmModel  ************************")
    print(model_input.shape)
    print(batch_size)

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
      
    ## Simple LSTM student
    simple_lstm = tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)


    loss = 0.0
    l2_penalty=1e-8
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(simple_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    #student_state = slim.fully_connected(state, 4096, activation_fn=None, weights_regularizer=slim.l2_regularizer(l2_penalty))
    student_state = state
    return student_state

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
    #zero = tf.cast(0, tf.int64)
    #num_frames_L1 = [tf.minimum(tf.cast(len_lower_lstm, tf.int64) , tf.maximum(zero, num_frames - int(len_lower_lstm) * i) ) for i in range(FLAGS.num_inputs_to_lstm)]
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



class SmoothHierarchicalLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

    # split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    outs, _ = tf.nn.dynamic_rnn(L1_stacked_lstm, model_input,
      sequence_length = num_frames, dtype = tf.float32)
    outs = tf.nn.relu(outs)
    outs_split = tf.split(outs, FLAGS.num_inputs_to_lstm, axis = 1)
    L2_input = []
    for i in range(FLAGS.num_inputs_to_lstm):
      outs_i_sum = tf.reduce_sum(outs_split[i], axis = 1)
      outs_i_mean = tf.transpose(tf.transpose(outs_i_sum) / (tf.cast(num_frames_L1[i], tf.float32) + 0.001))
      L2_input.append(outs_i_mean)
    L2_input = tf.stack(L2_input, axis = 1)
    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalResidualModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
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

    c1 = tf.Variable(0.99)
    c2 = tf.Variable(0.01)
    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      W2 = tf.get_variable('W', [state.get_shape()[-1], model_input.get_shape()[-1]])
      b2 = tf.get_variable('b', [model_input.get_shape()[-1]])
      state = tf.matmul(state, W2) + b2
      state = tf.nn.l2_normalize(state, -1)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      state = state*c1 + avg_inp*c2
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalDoubleResidualModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

    z1 = tf.Variable(0.99)
    z2 = tf.Variable(0.01)
    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        W1 = tf.get_variable('W1', [state.get_shape()[-1], model_input.get_shape()[-1]])
        b1 = tf.get_variable('b1', [model_input.get_shape()[-1]])
        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32) + 0.001))
        state = tf.matmul(state, W1) + b1
        state = tf.nn.relu(state)
        state = tf.nn.l2_normalize(state, -1)
        state = state * z1 + avg_inp * z2
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    c1 = tf.Variable(0.99)
    c2 = tf.Variable(0.01)
    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      W2 = tf.get_variable('W', [state.get_shape()[-1], model_input.get_shape()[-1]])
      b2 = tf.get_variable('b', [model_input.get_shape()[-1]])
      state = tf.matmul(state, W2) + b2
      state = tf.nn.l2_normalize(state, -1)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      state = state*c1 + avg_inp*c2
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
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
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalDoubleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

# Dropout Models===========================================

class BidirectionalQuadResidualConcatDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]
      l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l1_states = tf.reduce_max(l1_states, axis=1)

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l2_outputs = tf.reduce_max(l2_outputs, axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp, mean_l1_states, max_l1_states, mean_l2_outputs, max_l2_outputs], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalQuadResidualConcatDoubleDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)
    L2_input = tf.nn.dropout(L2_input, keep_prob=dropout)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]
      l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l1_states = tf.reduce_max(l1_states, axis=1)

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l2_outputs = tf.reduce_max(l2_outputs, axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp, mean_l1_states, max_l1_states, mean_l2_outputs, max_l2_outputs], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalDoubleResidualConcatDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]

      l2_states = tf.concat(states,axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp], axis = -1)
      state = tf.nn.dropout(state, keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalDoubleResidualConcatDoubleDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)
    L2_input = tf.nn.dropout(L2_input,keep_prob = dropout)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]

      l2_states = tf.concat(states,axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp], axis = -1)
      state = tf.nn.dropout(state, keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class DoubleConcatResidualConcatDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    with tf.variable_scope("LSTM1"):
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
      num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
      L1_outputs = []
      for i in range(FLAGS.num_inputs_to_lstm):
        with tf.variable_scope("RNN_L1") as scope:
          if i > 0:
            scope.reuse_variables()
          _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                          sequence_length = num_frames_L1[i],
                                          dtype = tf.float32)

          # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
          # max_inp = tf.reduce_max(split_model_input[i], axis = 1)          
          # state = tf.concat([state, avg_inp, max_inp], axis = -1)
          L1_outputs.append(state)

      L2_input = tf.stack(L1_outputs, axis = 1)

      with tf.variable_scope("RNN_L2"):
        _, state_enc1 = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                           sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                           dtype=tf.float32)

    with tf.variable_scope("LSTM2"):
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


      len_lower_lstm = FLAGS.num_inputs_to_lstm
      num_inputs_to_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm

      split_model_input = tf.split(model_input, num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
      num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(num_inputs_to_lstm)]
      L1_outputs = []
      for i in range(num_inputs_to_lstm):
        with tf.variable_scope("RNN_L1") as scope:
          if i > 0:
            scope.reuse_variables()
          _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                          sequence_length = num_frames_L1[i],
                                          dtype = tf.float32)

          # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
          # max_inp = tf.reduce_max(split_model_input[i], axis = 1)

          # state = tf.concat([state, avg_inp, max_inp], axis = -1)
          L1_outputs.append(state)

      L2_input = tf.stack(L1_outputs, axis = 1)

      with tf.variable_scope("RNN_L2"):
        _, state_enc2 = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                           sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                           dtype=tf.float32)

    avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
    input_mask = tf.cast(tf.sequence_mask(num_frames,FLAGS.max_num_frames),dtype=tf.float32)
    pre_stddev = tf.multiply(model_input - avg_inp[:,None,:], input_mask[:,:,None])
    stddev_inp =  tf.sqrt(tf.realdiv( tf.reduce_sum(tf.multiply(pre_stddev,pre_stddev),axis=1), tf.cast(num_frames[:,None], tf.float32) ))
    max_inp = tf.reduce_max(model_input, axis = 1)
    state = tf.concat([state_enc1, state_enc2, avg_inp, max_inp, stddev_inp], axis = -1)
    state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class TripleConcatResidualConcatDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    reuse=False
    states=[]
    for num_inputs_to_lstm in [10,15,25]:
      len_lower_lstm = FLAGS.max_num_frames/num_inputs_to_lstm
      split_model_input = tf.split(model_input, num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
      num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(num_inputs_to_lstm)]
      L1_outputs = []
      for i in range(num_inputs_to_lstm):
        with tf.variable_scope("RNN_L1") as scope:
          if i > 0 or reuse:
            scope.reuse_variables()
          _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                          sequence_length = num_frames_L1[i],
                                          dtype = tf.float32)

          # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
          # max_inp = tf.reduce_max(split_model_input[i], axis = 1)

          # state = tf.concat([state, avg_inp, max_inp], axis = -1)
          L1_outputs.append(state)

      L2_input = tf.stack(L1_outputs, axis = 1)

      with tf.variable_scope("RNN_L2",reuse=reuse):
        _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                           sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                           dtype=tf.float32)
      states.append(state[:,:lstm_size])
      reuse=True

    states = tf.concat(states,axis=1)
    avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
    input_mask = tf.cast(tf.sequence_mask(num_frames,FLAGS.max_num_frames),dtype=tf.float32)
    pre_stddev = tf.multiply(model_input - avg_inp[:,None,:], input_mask[:,:,None])
    stddev_inp =  tf.sqrt(tf.realdiv( tf.reduce_sum(tf.multiply(pre_stddev,pre_stddev),axis=1), tf.cast(num_frames[:,None], tf.float32) ))
    max_inp = tf.reduce_max(model_input, axis = 1)
    state = tf.concat([states, avg_inp, max_inp, stddev_inp], axis = -1)
    state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class L2MeanMaxResidualConcatDropout(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
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
    reuse=False
    states=[]
    for num_inputs_to_lstm in [10,15,25]:
      len_lower_lstm = FLAGS.max_num_frames/num_inputs_to_lstm
      split_model_input = tf.split(model_input, num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
      num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(num_inputs_to_lstm)]
      L1_outputs = []
      for i in range(num_inputs_to_lstm):
        with tf.variable_scope("RNN_L1") as scope:
          if i > 0 or reuse:
            scope.reuse_variables()
          _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                          sequence_length = num_frames_L1[i],
                                          dtype = tf.float32)

          # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
          # max_inp = tf.reduce_max(split_model_input[i], axis = 1)

          # state = tf.concat([state, avg_inp, max_inp], axis = -1)
          L1_outputs.append(state)


      L2_input = tf.stack(L1_outputs, axis = 1)
      with tf.variable_scope("RNN_L2",reuse=reuse):
        _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                           sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                           dtype=tf.float32)
      states.append(state[:,:lstm_size])
      reuse=True

    states = tf.stack(states,axis=1)
    states_mean = tf.reduce_mean(states,axis=1)
    states_max = tf.reduce_max(states,axis=1)
    avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
    input_mask = tf.cast(tf.sequence_mask(num_frames,FLAGS.max_num_frames),dtype=tf.float32)
    pre_stddev = tf.multiply(model_input - avg_inp[:,None,:], input_mask[:,:,None])
    stddev_inp =  tf.sqrt(tf.realdiv( tf.reduce_sum(tf.multiply(pre_stddev,pre_stddev),axis=1), tf.cast(num_frames[:,None], tf.float32) ))
    max_inp = tf.reduce_max(model_input, axis = 1)
    state = tf.concat([states_mean, states_max, avg_inp, max_inp, stddev_inp], axis = -1)
    state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalLowerTripleResidualConcatDropoutNoMax(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        state = tf.concat([state, avg_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]
      l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      l2_states = tf.concat(states,axis=1)
      # l2_outputs = tf.concat(outputs,axis=2)
      # sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      # mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      state = tf.concat([l2_states, avg_inp, mean_l1_states], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalUpperTripleResidualConcatDropoutNoMax(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        state = tf.concat([state, avg_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))

      # mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      # mask = mask[:,:,None]
      # l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      # sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      # mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      state = tf.concat([l2_states, avg_inp, mean_l2_outputs], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalTripleResidualConcatDropoutNoMax(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        # state = tf.concat([state, avg_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]
      l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      state = tf.concat([l2_states, avg_inp, mean_l1_states, mean_l2_outputs], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BidirectionalUpperDoubleResidualConcatDropoutNoMax(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        # state = tf.concat([state, avg_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))

      # mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      # mask = mask[:,:,None]
      # l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      # sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      # mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))

      state = tf.concat([l2_states, avg_inp, mean_l2_outputs], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class TripleHierBidirectionalUpperDoubleResidualConcatDropoutNoMax(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, dropout, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    with tf.variable_scope('L1'):
      L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    with tf.variable_scope('L2'):
      L2_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    with tf.variable_scope('forward_L3'):
      L3_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L3'):
      L3_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    bucket_size_1 = 5
    bucket_size_2 = 6
    bucket_size_3 = 10

    split_model_input = tf.split(model_input, bucket_size_2*bucket_size_3, axis = 1, name = 'lstm_l1_split')
    # len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm

    num_frames_L1 = [tf.minimum( bucket_size_1 , tf.maximum(0, num_frames - int(bucket_size_1) * i)) for i in range(bucket_size_2*bucket_size_3)]
    L1_outputs = []
    for i in range(bucket_size_2*bucket_size_3):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        # avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        # state = tf.concat([state, avg_inp], axis = -1)
        L1_outputs.append(state)
    L2_input = tf.stack(L1_outputs, axis = 1)
    L2_input = tf.split(L2_input, bucket_size_3, axis=1, name='lstm_l2_split')

    L2_seq_len_0 = [tf.minimum(int(bucket_size_1*bucket_size_2), tf.maximum(0, num_frames - int(bucket_size_1*bucket_size_2) * i)) for i in range(bucket_size_3)]
    L2_seq_len = [tf.cast(tf.ceil(tf.cast(L2_seq_len_0[i], tf.float32)/(bucket_size_1)),tf.int32) for i in range(bucket_size_3)]
    L2_outputs = []
    for i in range(bucket_size_3):
      with tf.variable_scope("RNN_L2") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input[i],
                                        sequence_length = L2_seq_len[i],
                                        dtype = tf.float32)
        L2_outputs.append(state)

    L3_input = tf.stack(L2_outputs, axis = 1)

    with tf.variable_scope("RNN_L3"):
      L3_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/(bucket_size_1*bucket_size_2)),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L3_stacked_lstm_fw, 
                    cell_bw = L3_stacked_lstm_bw, 
                    inputs = L3_input,
                    sequence_length=L3_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.realdiv(tf.reduce_sum(model_input, axis = 1) , tf.cast(num_frames, tf.float32)[:,None])

      l3_states = tf.concat(states,axis=1)
      l3_outputs = tf.concat(outputs,axis=2)
      sum_l3_outputs = tf.reduce_sum(l3_outputs, axis=1)
      mean_l3_outputs = tf.realdiv(sum_l3_outputs, tf.cast(L3_seq_len[:,None],dtype=tf.float32))

      state = tf.concat([l3_states, avg_inp, mean_l3_outputs], axis = -1)
      state = tf.nn.dropout(state,keep_prob = dropout)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)
# ============================================Dropout Models

class HierarchicalTripleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=L2_seq_len,
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      l1_states = tf.multiply(mask[:,:,None],L2_input[:,:,:2*lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l1_states = tf.reduce_max(l1_states, axis=1)
      state = tf.concat([state, avg_inp, max_inp,mean_l1_states,max_l1_states], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalBidirectionalDoubleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]

      l2_states = tf.concat(states,axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalBidirectionalUpperTripleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l2_outputs = tf.reduce_max(l2_outputs, axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp, mean_l2_outputs, max_l2_outputs], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalBidirectionalQuadResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]
      l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l1_states = tf.reduce_max(l1_states, axis=1)

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, tf.cast(L2_seq_len[:,None],dtype=tf.float32))
      max_l2_outputs = tf.reduce_max(l2_outputs, axis=1)

      state = tf.concat([l2_states, avg_inp, max_inp, mean_l1_states, max_l1_states, mean_l2_outputs, max_l2_outputs], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class PseudoEnsembleModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, labels, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    with tf.variable_scope('forward_L2'):
      L2_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
    with tf.variable_scope('backward_L2'):
      L2_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      L2_seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      L2_seq_len_float = tf.cast(L2_seq_len[:,None],dtype=tf.float32)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = L2_stacked_lstm_fw, 
                    cell_bw = L2_stacked_lstm_bw, 
                    inputs = L2_input,
                    sequence_length=L2_seq_len,
                    dtype=tf.float32)

      avg_inp = tf.realdiv(tf.reduce_sum(model_input, axis = 1), tf.cast(num_frames[:,None], tf.float32) )
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      # max_inp = tf.reduce_max(model_input, axis = 1)

      mask = tf.cast(tf.sequence_mask(L2_seq_len, FLAGS.num_inputs_to_lstm),dtype=tf.float32)
      mask = mask[:,:,None]
      l1_states = tf.multiply(mask,L2_input[:,:,:lstm_size])
      sum_l1_states = tf.reduce_sum(l1_states, axis=1)
      mean_l1_states = tf.realdiv(sum_l1_states, L2_seq_len_float)
      # max_l1_states = tf.reduce_max(l1_states, axis=1)

      l2_states = tf.concat(states,axis=1)
      l2_outputs = tf.concat(outputs,axis=2)
      sum_l2_outputs = tf.reduce_sum(l2_outputs, axis=1)
      mean_l2_outputs = tf.realdiv(sum_l2_outputs, L2_seq_len_float)
      # max_l2_outputs = tf.reduce_max(l2_outputs, axis=1)

    predictions = []
    l2_penalty=1e-8
    epsilon = 10e-6
    float_labels = tf.cast(labels, tf.float32)
    # loss = 0
    for step in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("last_layer") as scope:
        if(step>0):
          scope.reuse_variables()
        final_state = tf.layers.dense(
            tf.concat([l2_outputs[:,step,:],L2_input[:,step,:2*lstm_size+FLAGS.input_features],avg_inp],axis=1), 
            units=1024, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)        
        pred = slim.fully_connected(
            final_state, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        pred = tf.multiply(pred,mask[:,step,:])
        predictions.append(pred)

        # cross_entropy_loss = float_labels * tf.log(pred + epsilon) + (
        #     1 - float_labels) * tf.log(1 - pred + epsilon)
        # mask = tf.cast(tf.greater(L2_seq_len - step, 0),dtype=tf.float32)
        # loss += tf.reduce_sum(tf.multiply(cross_entropy_loss,mask[:,None]), 1)
    # final_loss = tf.reduce_mean(tf.realdiv(loss,tf.cast(L2_seq_len,dtype=tf.float32)))

    final_predictions = tf.reduce_max(tf.stack(predictions,axis=1),axis=1)
    return {"predictions": final_predictions}

    # aggregated_model = getattr(video_level_models,
    #                            FLAGS.video_level_classifier_model)
    # return aggregated_model().create_model(
    #     model_input=state,
    #     vocab_size=vocab_size,
    #     **unused_params)

class HierarchicalOverlappingDoubleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

    # split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    bucket_begin_indices = range(0,FLAGS.max_num_frames-len_lower_lstm+1,len_lower_lstm/2)
    split_model_input = [model_input[:,i:i+len_lower_lstm,:] for i in bucket_begin_indices]

    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - i)) for i in bucket_begin_indices]

    L1_outputs = []
    for i in range(len(split_model_input)):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(
                                          tf.minimum(
                                            tf.ceil(tf.cast(num_frames, tf.float32)/(len_lower_lstm/2)),
                                            2*FLAGS.num_inputs_to_lstm-1),
                                          tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalDifferenceDoubleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

    shape_inputs = tf.shape(model_input)
    diff_input = model_input - tf.concat(
        [ tf.zeros( shape=(shape_inputs[0],1,shape_inputs[2]), dtype=tf.float32 ),
        model_input[:,:-1,:] ]
        ,axis = 1
        )
    L1_input = tf.concat([model_input,diff_input],axis=2)
    split_model_input = tf.split(L1_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class HierarchicalNormedDoubleResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)

        avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(split_model_input[i], axis = 1)) / (tf.cast(num_frames_L1[i], tf.float32)+0.001))
        max_inp = tf.reduce_max(split_model_input[i], axis = 1)
        state = tf.concat([state, avg_inp, max_inp], axis = -1)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      # avg_inp_tile = tf.tile(avg_inp, [1,  , 1])
      # stddev_inp = (tf.transpose(tf.transpose(tf.reduce_sum((model_input - avg_inp)**2, axis = 1)) / tf.cast(num_frames, tf.float32))) ** 0.5
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
      mean,variance = tf.nn.moments(state,[0])
      state = tf.nn.batch_normalization(state,mean,variance,None,None,variance_epsilon=0.000001)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class SmoothHierarchicalResidualConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

    # split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    outs, _ = tf.nn.dynamic_rnn(L1_stacked_lstm, model_input,
      sequence_length = num_frames, dtype = tf.float32)
    outs = tf.nn.relu(outs)
    outs_split = tf.split(outs, FLAGS.num_inputs_to_lstm, axis = 1)
    L2_input = []
    for i in range(FLAGS.num_inputs_to_lstm):
      outs_i_sum = tf.reduce_sum(outs_split[i], axis = 1)
      outs_i_mean = tf.transpose(tf.transpose(outs_i_sum) / (tf.cast(num_frames_L1[i], tf.float32) + 0.001))
      L2_input.append(outs_i_mean)
    L2_input = tf.stack(L2_input, axis = 1)
    with tf.variable_scope("RNN_L2"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class InputAttentionRNNCell(RNNCell):
  def __init__(self,cell,state_is_tuple=True):
    super(InputAttentionRNNCell, self).__init__()
    self._cell = cell
    self._state_is_tuple = state_is_tuple
    self.reuse = False
  @property
  def state_size(self):
    return self._cell.state_size
  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self,inputs,state):
    with tf.variable_scope('attention',reuse=self.reuse):
      len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
      inputs = tf.reshape(inputs, [-1,len_lower_lstm,FLAGS.lstm_cells])
      hidden_size = FLAGS.lstm_cells

      state_hidden = tf.layers.dense(state,units=hidden_size,kernel_initializer = xavier)
      inputs_hidden = tf.layers.dense(inputs,units=hidden_size,kernel_initializer = xavier)
      hidden = tf.tanh(tf.add(state_hidden[:,None,:],inputs_hidden))
      
      attention_weights = tf.squeeze(tf.layers.dense(hidden,units = 1,kernel_initializer = xavier),[2])
      attention_weights = tf.nn.softmax(attention_weights)
      inputs = tf.reduce_sum(tf.multiply(attention_weights[:,:,None],inputs),axis=1)
      # attention_layer = tf.contrib.seq2seq.LuongAttention(num_units = self._cell.state_size, memory=inputs, normalize = True)
      # attention_values = attention_layer(state)   # [batch_size,memory_size]
      # attention_values = attention_values[:,:,None]
      # inputs = tf.reduce_sum(tf.multiply(inputs,attention_values),axis=1)
    self.reuse = True
    return self._cell(inputs,state)

class SmoothHierarchicalDoubleResidualDoubleConcatAttentionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    L2_stacked_lstm = InputAttentionRNNCell(L2_stacked_lstm,state_is_tuple=False)

    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    outs, _ = tf.nn.dynamic_rnn(L1_stacked_lstm, model_input,
      sequence_length = num_frames, dtype = tf.float32)
    outs = tf.concat([outs,model_input],axis=2)
    outs = tf.layers.dense(outs,units=lstm_size,activation = tf.tanh,kernel_initializer = xavier)

    L2_input = tf.reshape(outs,[-1,FLAGS.num_inputs_to_lstm,len_lower_lstm*lstm_size])
    with tf.variable_scope("RNN_L2"):
      outs_L2, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class SmoothHierarchicalDifferenceDoubleResidualDoubleConcatAttentionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    L2_stacked_lstm = InputAttentionRNNCell(L2_stacked_lstm,state_is_tuple=False)

    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    
    shape_inputs = tf.shape(model_input)
    L1_input = model_input - tf.concat(
        [ tf.zeros( shape=(shape_inputs[0],1,shape_inputs[2]), dtype=tf.float32 ),
        model_input[:,:-1,:] ]
        ,axis = 1
        )

    outs, _ = tf.nn.dynamic_rnn(L1_stacked_lstm, L1_input,
      sequence_length = num_frames, dtype = tf.float32)
    outs = tf.concat([outs,model_input],axis=2)
    outs = tf.layers.dense(outs,units=lstm_size,activation = tf.tanh,kernel_initializer = xavier)

    L2_input = tf.reshape(outs,[-1,FLAGS.num_inputs_to_lstm,len_lower_lstm*lstm_size])
    with tf.variable_scope("RNN_L2"):
      outs_L2, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class SmoothHierarchicalDifferenceDoubleResidualDoubleConcatModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    
    shape_inputs = tf.shape(model_input)
    L1_input = model_input - tf.concat(
        [ tf.zeros( shape=(shape_inputs[0],1,shape_inputs[2]), dtype=tf.float32 ),
        model_input[:,:-1,:] ]
        ,axis = 1
        )

    outs, _ = tf.nn.dynamic_rnn(L1_stacked_lstm, L1_input,
      sequence_length = num_frames, dtype = tf.float32)
    outs = tf.concat([outs,model_input],axis=2)

    L2_input = outs[:,len_lower_lstm - 1::len_lower_lstm,:]
    with tf.variable_scope("RNN_L2"):
      outs_L2, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32),
                                         dtype=tf.float32)
      avg_inp = tf.transpose(tf.transpose(tf.reduce_sum(model_input, axis = 1)) / tf.cast(num_frames, tf.float32))
      max_inp = tf.reduce_max(model_input, axis = 1)
      state = tf.concat([state, avg_inp, max_inp], axis = -1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class WindowDetectionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    # L1_stacked_lstm = tf.contrib.rnn.OutputProjectionWrapper(L1_stacked_lstm, vocab_size)


    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []


    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        W = tf.get_variable("RNN_W", shape = [state.get_shape()[-1], vocab_size])
        b = tf.get_variable("RNN_b", shape = [vocab_size])
        state_projected = tf.matmul(state, W) + b
        probabilities = tf.nn.sigmoid(state_projected)
        sum_state = tf.reduce_sum(state, axis = -1)
        L1_outputs.append(tf.transpose(tf.transpose(probabilities) * sum_state / (sum_state + 10.0**-8)))

    L1_outputs = tf.stack(L1_outputs, axis = 1)
    pred = tf.reduce_max(L1_outputs, axis = 1)

    return {"predictions": pred}

class HierarchicalWindowDetectionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    L1_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size/2, forget_bias=1.0, activation = tf.sigmoid)
                for _ in range(number_of_layers)
                ])

    L2_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size/2, forget_bias=1.0, activation = tf.tanh)
                for _ in range(number_of_layers)
                ])

    # L2_lstm = tf.contrib.rnn.OutputProjectionWrapper(L2_lstm, 512)

    split_model_input = tf.split(model_input, FLAGS.num_inputs_to_lstm, axis = 1, name = 'lstm_l1_split')
    len_lower_lstm = FLAGS.max_num_frames/FLAGS.num_inputs_to_lstm
    num_frames_L1 = [tf.minimum(int(len_lower_lstm), tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in range(FLAGS.num_inputs_to_lstm)]
    L1_outputs = []
    for i in range(FLAGS.num_inputs_to_lstm):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
          scope.reuse_variables()
        outs, _ = tf.nn.dynamic_rnn(L1_lstm, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        outs = tf.reduce_max(outs, axis = 1)
        # outs = tf.transpose(tf.transpose(outs) / (tf.cast(num_frames_L1[i], tf.float32)) + 0.001)
        L1_outputs.append(outs)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L2"):
      seq_len = tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/len_lower_lstm),tf.int32)
      outs, _ = tf.nn.dynamic_rnn(L2_lstm, L2_input,
                                         sequence_length=seq_len,
                                         dtype=tf.float32)
      # outs = tf.nn.relu(outs)
      # outs = tf.reduce_max(outs, axis = 1)
      # print tf.shape(outs)
      outs = tf.reshape(outs, [-1, 512])
      W = tf.get_variable("W", shape = [512, vocab_size])
      b = tf.get_variable("b", shape = [vocab_size])
      outs = tf.matmul(outs, W) + b
      outs = tf.sigmoid(outs)
      pred = tf.reshape(outs, [-1, FLAGS.num_inputs_to_lstm, vocab_size])
      pred = tf.reduce_max(pred, axis = 1)
      # outs = tf.reduce_sum(outs, axis = 1)
      # outs = tf.transpose(tf.transpose(outs) / (tf.cast(seq_len, tf.float32)) + 0.001)
      # outs = tf.nn.sigmoid(outs)
      # outs = outs * 2 - 1

    return {'predictions': pred}
    # aggregated_model = getattr(video_level_models,
    #                            FLAGS.video_level_classifier_model)
    # return aggregated_model().create_model(
    #     model_input=outs,
    #     vocab_size=vocab_size,
    #     **unused_params)

class PoorNoneAttentionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    num_features = model_input.get_shape().as_list()[2]

    L1_stacked_lstm_attention = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False, activation = tf.tanh)
                for _ in range(1)
                ],
            state_is_tuple=False)

    L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False, activation = tf.tanh)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    L2_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False, activation = tf.tanh)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    split_model_input = tf.split(model_input, 20, axis = 1, name = 'lstm_l1_split')
    num_frames_L1 = [tf.minimum(15, tf.maximum(0, num_frames - 15 * i)) for i in range(20)]
    
    L1_outputs = []
    W = tf.get_variable('attention_projection_weight',
                        [num_features, 1],
                        initializer=tf.contrib.layers.xavier_initializer())

    for i in range(20):
      with tf.variable_scope("RNN_L1") as scope:
        if i > 0:
            scope.reuse_variables()
        out, _ = tf.nn.dynamic_rnn(L1_stacked_lstm_attention, split_model_input[i],
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        out = tf.reshape(out, [-1, num_features])
        scalars = tf.reshape(tf.sigmoid(tf.matmul(out, W)), [-1, 15])
        scalar_sum = tf.tile(tf.reshape(tf.reduce_sum(scalars, axis = 1), [-1, 1]), [1,15])
        weights = scalars / scalar_sum
        weights = tf.tile(tf.reshape(weights, [-1, 15, 1]), [1,1,num_features])
        weighted_inp = weights * split_model_input[i]
        print(weighted_inp.get_shape().as_list())
      with tf.variable_scope("RNN_L2") as scope:
        if i > 0:
            scope.reuse_variables()
        _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, weighted_inp,
                                        sequence_length = num_frames_L1[i],
                                        dtype = tf.float32)
        L1_outputs.append(state)

    L2_input = tf.stack(L1_outputs, axis = 1)

    with tf.variable_scope("RNN_L3"):
      _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(tf.ceil(tf.cast(num_frames, tf.float32)/15.0),tf.int32),
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)
