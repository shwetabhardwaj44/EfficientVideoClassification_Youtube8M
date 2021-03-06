
"""Binary for generating predictions over a set of videos."""

import os
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

import glob
import numpy as np

FLAGS = flags.FLAGS

if __name__ == '__main__':
  num_classes = 4716
  
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from.")
  flags.DEFINE_string("output_file", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("preds_pattern", "",
                      "Filenames for prediction files")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", True,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 128,
      "How many examples to process per batch.")
  flags.DEFINE_float("dropout", 1.0,
                     "Dropout Probability")
  flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("device", "/gpu:0", "device on which we run inference.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 5,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")

def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
  #  print("Type - Test :")
  #  print(type(video_ids[video_index]))
  #  print(video_ids[video_index].decode('utf-8'))
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair
                                                  for pair in line) + "\n"


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
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
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            capacity=FLAGS.batch_size * 50,
                            allow_smaller_final_batch = True,
                            enqueue_many=True))
    return video_id_batch, video_batch, num_frames_batch

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess, gfile.Open(out_file_location, "w+") as out_file:
    with tf.device(FLAGS.device):
      video_id_batch, video_batch, num_frames_batch = get_input_data_tensors(reader, data_pattern, batch_size)
      latest_checkpoint = tf.train.latest_checkpoint(train_dir)
      if latest_checkpoint is None:
        raise Exception("unable to find a checkpoint at location: %s" % train_dir)
      else:
        meta_graph_location = latest_checkpoint + ".meta"
        logging.info("loading meta-graph: " + meta_graph_location)
      saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
      logging.info("restoring variables from " + latest_checkpoint)
      saver.restore(sess, latest_checkpoint)
      input_tensor = tf.get_collection("input_batch_raw")[0]
      num_frames_tensor = tf.get_collection("num_frames")[0]
      predictions_tensor = tf.get_collection("predictions")[0]
      old_predictions = tf.get_collection("old_predictions")[0]
      update_dropout_test = tf.get_collection("update_dropout_test")
      if(len(update_dropout_test) > 0):
        update_dropout_test = update_dropout_test[0]
      else:
        update_dropout_test = None

      # Workaround for num_epochs issue.
      def set_up_init_ops(variables):
        init_op_list = []
        for variable in list(variables):
          if "train_input" in variable.name:
            init_op_list.append(tf.assign(variable, 1))
            variables.remove(variable)
        init_op_list.append(tf.variables_initializer(variables))
        return init_op_list

      sess.run(set_up_init_ops(tf.get_collection_ref(
          tf.GraphKeys.LOCAL_VARIABLES)))
      # Updating dropout keep_prob to 1.0
      if(update_dropout_test is not None):
        sess.run(update_dropout_test)
        logging.info('Updated dropout')

      def read_pred_file(filename):
        logging.info('reading: '+filename)
        f = open(filename).readlines()
        f = f[1:]
        f = map(lambda x: x.split(','),f)
        video_ids = [x[0] for x in f]

        preds  = [x[1].split(' ') for x in f]
        labels = [map(int,x[0::2]) for x in preds]
        probs  = [map(float,x[1::2]) for x in preds]

        d = {video_ids[i]: (labels[i],probs[i]) for i in range(len(video_ids))}
        return d

      preds_files = glob.glob(FLAGS.preds_pattern)
      preds_files.sort()
      num_preds_files = len(preds_files)
      logging.info('files: '+' '.join(preds_files))

      predictions_dict = []
      for i in range(len(preds_files)):
        predictions_dict.append(read_pred_file(preds_files[i]))

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      num_examples_processed = 0
      start_time = time.time()
      out_file.write("VideoId,LabelConfidencePairs\n")

      try:
        while not coord.should_stop():
          video_id_batch_val, video_batch_val,num_frames_batch_val = sess.run([video_id_batch, video_batch, num_frames_batch])
          old_predictions_val = np.zeros(shape=(FLAGS.batch_size, num_preds_files, 4716),dtype=np.float32)
          for i,video_id in enumerate(video_id_batch_val):
            for j in range(num_preds_files):
              cur_labels,cur_probs = predictions_dict[j][video_id]
              old_predictions_val[i,j,cur_labels] = cur_probs
          now_1 = time.time()
          predictions_val, = sess.run([predictions_tensor], feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val, old_predictions: old_predictions_val})
          now = time.time()
          num_examples_processed += len(video_batch_val)
          num_classes = predictions_val.shape[1]
          for line in format_lines(video_id_batch_val, predictions_val, top_k):
            out_file.write(line)
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds (data): " + "{0:.2f}".format(now_1-start_time) + " (computation) " + "{0:.2f}".format(now-start_time))
        out_file.flush()


      except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
      finally:
        coord.request_stop()

      out_file.close()
      coord.join(threads)
      sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  with tf.device(FLAGS.device):
    inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
      FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
