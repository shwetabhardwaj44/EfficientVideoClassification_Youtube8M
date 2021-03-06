
"""Binary for PARALLEL training of Dyamic Teacher And Student Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

import eval_util
import losses
import frame_level_models
import video_level_models
import readers
import random 
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.ops import variables as tf_variables
import utils
import numpy as np
import cPickle as pkl

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", True,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_bool(
      "bagging", False,
      "Bagging.")

  flags.DEFINE_string(
      "model", "HierarchicalLstmModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch for training.")
  flags.DEFINE_integer("every_n", 1,
                       "every nth frame to be used by student.")

  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float("dropout", 0.5,
                     "Dropout Probability")
  flags.DEFINE_float(
      "regularization_penalty", 2,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.001,
                     "Which learning rate to start with.")
  flags.DEFINE_float("learning_rate_decay", 1,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")
  flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")
  flags.DEFINE_integer("num_epochs", 10,
                       "How many passes to make over the dataset before "
                       "halting training.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 4,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_integer("gpu", 0, "GPU on which the code will run")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    num_files = len(files)
    logging.info("Number of training files: %s.", str(num_files))
    if(FLAGS.bagging):
      np.random.seed( int(time.time()*10000)%1000000 )
      files = list(np.random.choice(files,num_files,replace=True))
      logging.info("Bagging done")
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=FLAGS.batch_size * 50,
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """
  
  global_step = tf.Variable(0, trainable=False, name="global_step")
  learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step * batch_size,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  
  global_step_stud = global_step
  learning_rate_stud = tf.train.exponential_decay(
      base_learning_rate,
      global_step * batch_size,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)

  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('learning_rate_stud', learning_rate_stud)

  optimizer = optimizer_class(learning_rate)
  optimizer_student = optimizer_class(learning_rate_stud)

  unused_video_id, model_input_raw, labels_batch, num_frames = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)
  
  feature_dim = len(model_input_raw.get_shape()) - 1

  ########## Input For Teacher ##########
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
  inp_dims =   model_input.get_shape().as_list()


  ########## Input For Student ##########

  max_num_frames_before_sampling = 300
  max_num_frames_student = int(max_num_frames_before_sampling/FLAGS.every_n)
  num_frames_student = tf.cast(tf.multiply(tf.divide(num_frames, max_num_frames_before_sampling), max_num_frames_student), tf.int64)
  frame_index = 0
  list_index_to_retain = []
  while(FLAGS.every_n*frame_index <= 299):
      list_index_to_retain.append(FLAGS.every_n*frame_index)
      frame_index += 1    
  obj = tf.transpose(model_input,[1,0,2])
  reduced_input = tf.gather(obj, list_index_to_retain)
  model_input_student = tf.transpose(reduced_input,[1,0,2])
  

  ##########################
  #### Define Teacher Model
  ##########################  
  dropout_var = tf.Variable(initial_value=FLAGS.dropout,trainable=False,expected_shape=[],dtype=tf.float32)
  update_dropout_test = dropout_var.assign(1.0)

  with tf.variable_scope("model"):
    teacher_state, result   = model.create_model(
        model_input,
        num_frames=num_frames,
        vocab_size=reader.num_classes,
        batch_size=FLAGS.batch_size,
        labels=labels_batch,
        dropout = dropout_var)

    predictions = result["predictions"]
    print("Confirming Shapes of batch_size, predictions and labels_batch:")
    print(FLAGS.batch_size, predictions.shape, labels_batch.shape)

    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)
    tf.summary.scalar("label_loss", label_loss)

    if "regularization_loss" in result.keys():
      reg_loss = result["regularization_loss"]
    else:
      reg_loss = tf.constant(0.0)
    
    reg_losses = tf.losses.get_regularization_losses(scope='model')
    if reg_losses:
      reg_loss += tf.add_n(reg_losses)
    
    if regularization_penalty != 0:
      tf.summary.scalar("reg_loss", reg_loss)

    # Adds update_ops (e.g., moving average updates in batch normalization) as
    # a dependency to the train_op.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if "update_ops" in result.keys():
      update_ops += result["update_ops"]
    if update_ops:
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name="gradient_barrier")
        with tf.control_dependencies([barrier]):
          label_loss = tf.identity(label_loss)

    # Incorporate the L2 weight penalties etc.
    final_loss = regularization_penalty * reg_loss + label_loss
    variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')
    names_variables_to_train = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')]
    print("Trainable Parameters of Teacher:")
    print(names_variables_to_train) 
    train_op = slim.learning.create_train_op(
        final_loss,
        optimizer,
        global_step=global_step,
        variables_to_train=variables_to_train,        
        clip_gradient_norm=clip_gradient_norm)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", predictions)
    tf.add_to_collection("input_batch_raw", model_input_raw)
    tf.add_to_collection("input_batch", model_input)
    tf.add_to_collection("update_dropout_test", update_dropout_test)
    tf.add_to_collection("num_frames", num_frames)
    tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("train_op", train_op)
  
  ##########################
  #### Define Student Model
  ##########################    
  with tf.variable_scope("model_student"):
    student_state, student_results  = model.create_model_inference(model_input_student,
        num_frames=num_frames_student,
        vocab_size=reader.num_classes,
        batch_size=FLAGS.batch_size,
        labels=labels_batch,
        every_n= FLAGS.every_n,
        num_inputs_L1 = 5,
        dropout = dropout_var)

    student_loss_state_batch = tf.reduce_sum(tf.square(tf.subtract(teacher_state, student_state)), axis= 1)
    print("Shape of student_loss_state_batch:")
    print(student_loss_state_batch.shape)
    student_loss_state = tf.reduce_mean(student_loss_state_batch)
    tf.summary.scalar("State_student_loss", student_loss_state)

    student_predictions = student_results["predictions"]
    print("Confirming Shapes of batch_size, predictions and labels_batch:")
    print(FLAGS.batch_size, student_predictions.shape, labels_batch.shape)

    if "loss" in student_results.keys():
      student_label_loss = student_results["loss"]
    else:
      student_label_loss = label_loss_fn.calculate_loss(student_predictions, labels_batch)
    tf.summary.scalar("student_label_loss", student_label_loss)

    if "regularization_loss" in student_results.keys():
      stud_reg_loss = student_results["regularization_loss"]
    else:
      stud_reg_loss = tf.constant(0.0)
    
    stud_reg_losses = tf.losses.get_regularization_losses(scope='model_student')
    if stud_reg_losses:
      stud_reg_loss += tf.add_n(stud_reg_losses)
    
    if regularization_penalty != 0:
      tf.summary.scalar("stud_reg_loss", stud_reg_loss)

    # Adds update_ops (e.g., moving average updates in batch normalization) as
    # a dependency to the train_op.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if "update_ops" in student_results.keys():
      update_ops += student_results["update_ops"]
    if update_ops:
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name="gradient_barrier")
        with tf.control_dependencies([barrier]):
          student_label_loss = tf.identity(student_label_loss)

    def KL_div(x, y):
        X = tf.distributions.Categorical(probs=x)
        Y = tf.distributions.Categorical(probs=y)
        return tf.distributions.kl_divergence(X, Y)
    pred_loss = tf.reduce_sum(KL_div(predictions, student_predictions))

    # Incorporate the L2 weight penalties etc.
    ### TOTAL_LOSS     = L_REP (student_loss_state) + L_PRED (pred_loss) + L_CE (student_label_loss) + REG_LOSS
    total_student_loss =  student_loss_state + pred_loss + student_label_loss + student_loss_state +  regularization_penalty * stud_reg_loss
    
    student_vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model_student')
    names_student_vars_to_train = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model_student')]
    print("Trainable Parameters of Student:")
    print(names_student_vars_to_train) 
    
    train_student_op = slim.learning.create_train_op(
        total_student_loss,
        optimizer_student,
        global_step=global_step_stud,
        variables_to_train=student_vars_to_train,        
        clip_gradient_norm=clip_gradient_norm)
    
    tf.add_to_collection("student_loss_state", student_loss_state)
    tf.add_to_collection("pred_loss", pred_loss)
    tf.add_to_collection("student_label_loss", student_label_loss)
    tf.add_to_collection("num_frames_student", num_frames_student)    
    tf.add_to_collection("train_student_op", train_student_op)    
    tf.add_to_collection("total_student_loss", total_student_loss)
    for variable in slim.get_model_variables():
      tf.summary.histogram(variable.op.name, variable)
  

class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, log_device_placement=True):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(log_device_placement=log_device_placement,
                                 allow_soft_placement=True)
    self.config.gpu_options.allow_growth = True

    if self.is_master and self.task.index > 0:
      raise StandardError("%s: Only one replica of master expected",
                          task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    #########################################
    # Remove training directory manually only
    #########################################

    # if self.is_master and start_new_model:
    #   self.remove_training_directory(self.train_dir)

    start_time = time.time()
    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)
    with tf.Graph().as_default() as graph:


      if meta_filename:
        saver = self.recover_model(meta_filename)

      # Original code for distributed training
      #with tf.device(device_fn):
      with tf.device("/gpu:%d"%FLAGS.gpu):

        if not meta_filename:
          saver = self.build_model()

        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_op = tf.get_collection("train_op")[0]
        train_student_op = tf.get_collection("train_student_op")[0]
        total_student_loss = tf.get_collection("total_student_loss")[0]
        student_loss_state = tf.get_collection("student_loss_state")[0]
        pred_loss = tf.get_collection("pred_loss")[0]
        student_label_loss = tf.get_collection("student_label_loss")[0]
        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=30 * 60,
        save_summaries_secs=120,
        saver=saver)

    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:

      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while not sv.should_stop():

          batch_start_time = time.time()

          # Parallel Training of Dynamic Teacher and Student network::
          _, _, global_step_val, loss_val, predictions_val, labels_val, total_stud_loss, stud_state_loss, stud_pred_loss, stud_label_loss = sess.run(
              [train_op, train_student_op, global_step, loss, predictions, labels, total_student_loss, student_loss_state, pred_loss, student_label_loss] )
          seconds_per_batch = time.time() - batch_start_time
      
          if self.is_master:
            examples_per_second = labels_val.shape[0] / seconds_per_batch
            hit_at_one = eval_util.calculate_hit_at_one(predictions_val,
                                                        labels_val)
            perr = eval_util.calculate_precision_at_equal_recall_rate(
                predictions_val, labels_val)
            gap = eval_util.calculate_gap(predictions_val, labels_val)

            logging.info(
                "%s: training step " + str(global_step_val) + "| Hit@1: " +
                ("%.2f" % hit_at_one) + "| PERR: " + ("%.2f" % perr) + "| GAP: " +
                ("%.2f" % gap) + "| Teacher_Loss: " + str(loss_val.round(2))+ "| L_REP: " + str(stud_state_loss.round(2)) +
                "| L_PRED: " + str(stud_pred_loss.round(2))+ "| L_CE: " + str(stud_label_loss.round(2)),
                task_as_string(self.task))

            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Perr", perr), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_GAP", gap), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("global_step/Examples/Second",
                                  examples_per_second), global_step_val)
            sv.summary_writer.flush()

      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()
    total_time = time.time() - start_time
    print('Total time taken is '+str(total_time))

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None
    
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint: 
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None
    
    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self):
    """Find the model and build the graph."""

    with tf.device("/gpu:%d"%FLAGS.gpu):

      # Convert feature_names and feature_sizes to lists of values.
      feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

      if FLAGS.frame_features:
        reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names, feature_sizes=feature_sizes)
      else:
        reader = readers.YT8MAggregatedFeatureReader(
          feature_names=feature_names, feature_sizes=feature_sizes)

      # Find the model.
      model = find_class_by_name(FLAGS.model,
                               [frame_level_models, video_level_models])()
      label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
      optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

      build_graph(reader=reader,
                 model=model,
                 optimizer_class=optimizer_class,
                 clip_gradient_norm=FLAGS.clip_gradient_norm,
                 train_data_pattern=FLAGS.train_data_pattern,
                 label_loss_fn=label_loss_fn,
                 base_learning_rate=FLAGS.base_learning_rate,
                 learning_rate_decay=FLAGS.learning_rate_decay,
                 learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                 regularization_penalty=FLAGS.regularization_penalty,
                 num_readers=FLAGS.num_readers,
                 batch_size=FLAGS.batch_size,
                 num_epochs=FLAGS.num_epochs)

      logging.info("%s: Built graph.", task_as_string(self.task))
      vars_models = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      names_vars_models = [v.name for v in vars_models]
 
      return tf.train.Saver(max_to_keep=1)


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Print out the flags.
  for k, v in tf.flags.FLAGS.__flags.iteritems():
    print("Key: %s Value: %s"%(k,v))
    
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    Trainer(cluster, task, FLAGS.train_dir, FLAGS.log_device_placement).run(
        start_new_model=FLAGS.start_new_model)
  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))


if __name__ == "__main__":
  app.run()
