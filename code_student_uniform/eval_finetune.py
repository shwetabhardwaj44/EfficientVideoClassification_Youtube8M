"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import time
import numpy as np
import eval_util
import losses
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from. "
                      "The tensorboard metrics files are also saved to this "
                      "directory.")
  flags.DEFINE_string(
      "eval_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", True,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "HierarchicalLstmModel",
      "Which architecture to use for the model. Options include 'Logistic', "
      "'SingleMixtureMoe', and 'TwoLayerSigmoid'. See aggregated_models.py and "
      "frame_level_models.py for the model definitions.")
  flags.DEFINE_integer("batch_size", 128,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("every_n", 1,
                       "every nth frame for student network.")  

  flags.DEFINE_integer("gpu", 0,
                       "Gpu on which to run eval.")
  
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Loss computed on validation data")
  # Other flags.
  flags.DEFINE_integer("num_readers", 4,
                       "How many threads to use for reading input files.")
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(reader,
                model,
                eval_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
	"""Creates the Tensorflow graph for evaluation.

	Args:
	reader: The data file reader. It should inherit from BaseReader.
	model: The core model (e.g. logistic or neural net). It should inherit
	       from BaseModel.
	eval_data_pattern: glob path to the evaluation data files.
	label_loss_fn: What kind of loss to apply to the model. It should inherit
	            from BaseLoss.
	batch_size: How many examples to process at a time.
	num_readers: How many threads to use for I/O operations.
	"""
	global_step = tf.Variable(0, trainable=False, name="global_step")
	video_id_batch, model_input_raw, labels_batch, num_frames_before_sampling = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
	  reader,
	  eval_data_pattern,
	  batch_size=batch_size,
	  num_readers=num_readers)
	tf.summary.histogram("model_input_raw", model_input_raw)

	feature_dim = len(model_input_raw.get_shape()) - 1

	# Normalize input features.
	model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
	# Modify input at evaluation
	max_num_frames_before_sampling = 300
	max_num_frames_student = int(max_num_frames_before_sampling/FLAGS.every_n)
	num_frames = tf.cast(tf.multiply(tf.divide(num_frames_before_sampling, max_num_frames_before_sampling), max_num_frames_student), tf.int64)
	frame_index = 0
	list_index_to_retain = []
	while(FLAGS.every_n*frame_index <= 299):
	  list_index_to_retain.append(FLAGS.every_n*frame_index)
	  frame_index += 1    
	obj = tf.transpose(model_input,[1,0,2])
	reduced_input = tf.gather(obj, list_index_to_retain)
	model_input_student = tf.transpose(reduced_input,[1,0,2])

	# We restore the parameters of 'model_student'
	with tf.variable_scope("model_student"):
		student_state, result = model.create_model_inference(model_input_student,
	                            num_frames=num_frames,
	                            vocab_size=reader.num_classes,
	                            batch_size=FLAGS.batch_size,
                              every_n = FLAGS.every_n,
	                            labels=labels_batch,
                              num_inputs_L1 = 5,
	                            is_training=False)
		predictions = result["predictions"]
		tf.summary.histogram("model_activations", predictions)
		if "loss" in result.keys():
		  label_loss = result["loss"]
		else:
		  label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

		tf.add_to_collection("global_step", global_step)
		tf.add_to_collection("loss", label_loss)
		tf.add_to_collection("predictions", predictions)
		tf.add_to_collection("input_batch", model_input_student)
		tf.add_to_collection("video_id_batch", video_id_batch)
		tf.add_to_collection("num_frames", num_frames)
		tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
		tf.add_to_collection("summary_op", tf.summary.merge_all())


def evaluation_loop(video_id_batch, prediction_batch, label_batch, loss,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val):
  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """
  
  global_step_val = -1
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)

    #latest_checkpoint = str(FLAGS.train_dir)+'model.ckpt-36704'
    if latest_checkpoint:

      logging.info("Loading checkpoint for eval: " + latest_checkpoint)
      # Restores from checkpoint
      saver.restore(sess, latest_checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-fav orite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = latest_checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, label_batch, loss, summary_op]
    coord = tf.train.Coordinator()
    total_example_per_sec = []
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()
        _, predictions_val, labels_val, loss_val, summary_val = sess.run(
            fetches)
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = labels_val.shape[0] / seconds_per_batch
        total_example_per_sec.append(example_per_second)
        examples_processed += labels_val.shape[0]

        iteration_info_dict = evl_metrics.accumulate(predictions_val,
                                                     labels_val, loss_val)
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      avg_example_per_sec = np.sum(np.asarray(total_example_per_sec))/len(total_example_per_sec)
      logging.info("Average examples processed in one second %0.20f" % avg_example_per_sec)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
  start_time = time.time() 
  with tf.device("/gpu:%d"%FLAGS.gpu):  
    tf.set_random_seed(0)  # for reproducibility
    with tf.Graph().as_default():
      # convert feature_names and feature_sizes to lists of values
      feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
          FLAGS.feature_names, FLAGS.feature_sizes)
  
      if FLAGS.frame_features:
        reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                                feature_sizes=feature_sizes)
      else:
        reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                     feature_sizes=feature_sizes)

      model = find_class_by_name(FLAGS.model,
           [frame_level_models, video_level_models])()
      label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()

      if FLAGS.eval_data_pattern is "":
        raise IOError("'eval_data_pattern' was not specified. " +
                     "Nothing to evaluate.")

      build_graph(
        reader=reader,
        model=model,
        eval_data_pattern=FLAGS.eval_data_pattern,
        label_loss_fn=label_loss_fn,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size)
      logging.info("built evaluation graph")
      video_id_batch = tf.get_collection("video_id_batch")[0]
      prediction_batch = tf.get_collection("predictions")[0]
      label_batch = tf.get_collection("labels")[0]
      loss = tf.get_collection("loss")[0]
      summary_op = tf.get_collection("summary_op")[0]

      # restore model_student into model
      vars_models = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_student')
      names_vars_models = [v.name for v in vars_models]
      logging.info("Names of trainable parameters in student:")
      logging.info(names_vars_models)
    
      saver = tf.train.Saver(tf.global_variables())
      summary_writer = tf.summary.FileWriter(
          FLAGS.train_dir, graph=tf.get_default_graph())

      evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k)

      last_global_step_val = -1
      while True:
        last_global_step_val = evaluation_loop(video_id_batch, prediction_batch,
                                               label_batch, loss, summary_op,
                                               saver, summary_writer, evl_metrics,
                                               last_global_step_val)
        if FLAGS.run_once:
          break
  total_time = time.time()-start_time
  print("Total time taken is "+str(total_time))


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()

