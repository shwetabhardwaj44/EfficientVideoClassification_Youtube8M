

"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf
import numpy as np

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLossWithSparsity(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)

      sparsity_reg = 0.1*tf.reduce_mean(tf.reduce_sum(predictions,axis=1))
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1)) + sparsity_reg

class CrossEntropyLossTop50(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      
      values,_ = tf.nn.top_k(predictions,k=50,sorted=True)
      values = values[:,49]
      mask = tf.cast(predictions>=values[:,None],dtype=tf.float32)

      cross_entropy_loss = tf.multiply(cross_entropy_loss,mask)*(4716.0/50.0)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

class PWELoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      labels = tf.cast(labels,tf.float32)

      predictions = tf.reshape(predictions,shape=[128,4716])
      labels = tf.reshape(labels,shape=[128,4716])

      predictions_ = tf.unstack(predictions,axis=0)
      labels_ = tf.unstack(labels,axis=0)
      loss = 0.0

      for i,j in zip(predictions_,labels_):
        pn_pairs = tf.matmul(i[:,None],(1-i)[None,:])
        opon_pairs = j[:,None] - j[None,:]
        exp_opon_pairs = tf.exp(-opon_pairs)
        inside = pn_pairs*exp_opon_pairs
        loss += inside/(tf.reduce_sum(i)*tf.reduce_sum(1-i))

      return loss/128.0

class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

class CrossEntropyLossClassImbalance(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      counts = map(int,open('counts_tv','r').readlines())

      positive_freq = [i/(4906660.0+1401828.0) for i in counts]
      positive_freq = np.sqrt(positive_freq)

      positive_weights = 1/positive_freq

      pw = tf.Variable(positive_weights,dtype=tf.float32,trainable=False)

      cross_entropy_loss = pw[None,:] * float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

class CrossEntropyLossPositives(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

class NewLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      bad_positive = tf.cast((predictions<0.9),dtype=tf.float32)
      cross_entropy_loss_1 = bad_positive * float_labels * tf.log(predictions + epsilon) 

      prob_positive = tf.multiply(predictions,float_labels) + (1-float_labels)
      min_prob_positive = tf.maximum(tf.reduce_min(prob_positive)-0.1,0.1)

      prob_negative = tf.multiply(predictions,1-float_labels)
      bad_negative = tf.cast((prob_negative>min_prob_positive),dtype = tf.float32)
      cross_entropy_loss_0 = bad_negative*(1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss_0+cross_entropy_loss_1)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

class HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.

  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
          all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class SoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 10e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)
