"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "num_hidden_units", 1024,
    "Number of hidden units.")

class SingleHiddenLayerModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-7, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # model_input = tf.concat([model_input[:,0:1024],model_input[:,2048:2048+1024]], axis=1)
    hidden = slim.fully_connected(
        model_input, FLAGS.num_hidden_units, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_regularizer=slim.l2_regularizer(1e-4))
    output = slim.fully_connected(
        hidden, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_regularizer=slim.l2_regularizer(1e-3))

    return {"predictions": output}

class SingleHiddenLayerModelDropout(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, dropout, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # model_input = tf.concat([model_input[:,0:1024],model_input[:,2048:2048+1024]], axis=1)
    hidden = slim.fully_connected(
        model_input, FLAGS.num_hidden_units, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hidden = tf.nn.dropout(hidden, keep_prob = dropout)
    output = slim.fully_connected(
        hidden, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_regularizer=slim.l2_regularizer(1e-4))

    return {"predictions": output}

class DoubleHiddenLayerModelDropout(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, dropout, l2_penalty=1e-8, **unused_params):
    # """Creates a logistic model.

    # model_input = tf.concat([model_input[:,0:1024],model_input[:,2048:2048+1024]], axis=1)
    # Hidden 1
    hidden = slim.fully_connected(
        model_input, 2048, activation_fn=tf.tanh,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hidden = tf.nn.dropout(hidden, keep_prob = dropout)

    # Hidden 2
    hidden = slim.fully_connected(
        hidden, 1024, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))

    output = slim.fully_connected(
        hidden, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_regularizer=slim.l2_regularizer(1e-4))

    return {"predictions": output}

class SplitSingleHiddenLayerModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, labels, l2_penalty=1e-7, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # model_input = tf.concat([model_input[:,0:1024],model_input[:,2048:2048+1024]], axis=1)
    loss = 0.0
    epsilon = 10e-6
    float_labels = tf.cast(labels, tf.float32)
    with tf.variable_scope('0-30'): # class count: 1,00,000+
      hidden = slim.fully_connected(
          model_input, 1024, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      output_0 = slim.fully_connected(
          hidden, 30, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))

      cross_entropy_loss = float_labels[:,:30] * tf.log(output_0 + epsilon) + (
          1 - float_labels[:,:30]) * tf.log(1 - output_0 + epsilon)
      cross_entropy_loss = tf.reduce_sum(tf.negative(cross_entropy_loss),axis=1)
      loss+=cross_entropy_loss

    with tf.variable_scope('30-300'): # class count: 10,000+
      hidden = slim.fully_connected(
          model_input, 512, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      output_1 = slim.fully_connected(
          hidden, 300-30, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      cross_entropy_loss = 2 * float_labels[:,30:300] * tf.log(output_1 + epsilon) + 0.25*(
          1 - float_labels[:,30:300]) * tf.log(1 - output_1 + epsilon)
      cross_entropy_loss = tf.reduce_sum(tf.negative(cross_entropy_loss),axis=1)
      loss+=cross_entropy_loss

    with tf.variable_scope('300-1500'): # class count: 1,000+
      hidden = slim.fully_connected(
          model_input, 256, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      output_2 = slim.fully_connected(
          hidden, 1500-300, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      cross_entropy_loss = 4 * float_labels[:,300:1500] * tf.log(output_2 + epsilon) + 0.1*(
          1 - float_labels[:,300:1500]) * tf.log(1 - output_2 + epsilon)
      cross_entropy_loss = tf.reduce_sum(tf.negative(cross_entropy_loss),axis=1)
      loss+=cross_entropy_loss

    with tf.variable_scope('1500-4716'): # class count: 100+
      hidden = slim.fully_connected(
          model_input, 256, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      output_3 = slim.fully_connected(
          hidden, 4716-1500, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      cross_entropy_loss = 10 * float_labels[:,1500:4716] * tf.log(output_3 + epsilon) + 0.01*(
          1 - float_labels[:,1500:4716]) * tf.log(1 - output_3 + epsilon)
      cross_entropy_loss = tf.reduce_sum(tf.negative(cross_entropy_loss),axis=1)
      loss+=cross_entropy_loss

    output = tf.concat([output_0,output_1,output_2,output_3],axis=1)

    return {"predictions": output, "loss": tf.reduce_mean(loss)}

class ScaledSingleHiddenLayerModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # model_input = tf.concat([model_input[:,0:1024],model_input[:,2048:2048+1024]], axis=1)
    hidden = slim.fully_connected(
        model_input, FLAGS.num_hidden_units, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        hidden, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_regularizer=slim.l2_regularizer(1e-3))

    o_max = tf.reduce_max(output,axis=1,keep_dims=True)
    o_min = tf.reduce_min(output,axis=1,keep_dims=True)

    output = tf.realdiv(output - o_min, o_max-o_min)

    return {"predictions": output}

class SingleHiddenLayerResidualModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, dropout, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    old_predictions = model_input[:,6400:]
    model_input = model_input[:,:6400]
    hidden = slim.fully_connected(
        model_input, FLAGS.num_hidden_units, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hidden = tf.nn.dropout(hidden,keep_prob = dropout)
    output = slim.fully_connected(
        hidden, vocab_size, activation_fn=tf.tanh,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_regularizer=slim.l2_regularizer(1e-4))
    output = tf.nn.sigmoid(tf.add(output,old_predictions))
    return {"predictions": output}

class LinearRegressionEnsemble(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, old_predictions, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # num_models = tf.shape(old_predictions)[1]
    old_predictions = old_predictions[:tf.shape(model_input)[0],:,:]
    weights = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.identity,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    weights = tf.nn.softmax(weights, dim = -1)

    output = tf.multiply(weights[:,:,None], old_predictions)
    output = tf.reduce_sum(output, axis = 1)
    
    return {"predictions": output}

class LinearRegressionWeightedMeanEnsemble(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, old_predictions, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # num_models = tf.shape(old_predictions)[1]
    old_predictions = old_predictions[:tf.shape(model_input)[0],:,:]
    weights = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.identity,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    weights = tf.nn.softmax(weights, dim = -1)

    output = tf.multiply(weights[:,:,None], old_predictions)
    output = tf.reduce_sum(output, axis = 1)
    
    alpha = tf.sigmoid(tf.Variable(2.0,trainable=True))
    mean_pred = tf.reduce_mean(old_predictions,axis=1)
    output = output * (1-alpha) + mean_pred * alpha

    return {"predictions": output}

class LinearRegressionWeightedMeanEnsembleDifferenceLoss(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, old_predictions, labels, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # num_models = tf.shape(old_predictions)[1]
    old_predictions = old_predictions[:tf.shape(model_input)[0],:,:]
    weights = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.identity,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    weights = tf.nn.softmax(weights, dim = -1)

    output = tf.multiply(weights[:,:,None], old_predictions)
    output = tf.reduce_sum(output, axis = 1)
    
    alpha = tf.sigmoid(tf.Variable(2.0,trainable=True))
    mean_pred = tf.reduce_mean(old_predictions,axis=1)
    output = output * (1-alpha) + mean_pred * alpha

    loss = - tf.multiply(output - mean_pred, labels) + tf.multiply(output - mean_pred, 1-labels)

    values,_ = tf.nn.top_k(output,k=20,sorted=True)
    values = values[:,19]
    mask = tf.cast(output>=values[:,None],dtype=tf.float32)
    loss = tf.multiply(loss,mask)
    return {"predictions": output, "loss": loss}

class IndependentClassEnsemble(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, old_predictions, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # num_models = tf.shape(old_predictions)[1]
    old_predictions = old_predictions[:tf.shape(model_input)[0],:,:]
    mask = tf.cast(old_predictions>0, dtype=tf.float32)

    weights = tf.get_variable(name='weights',shape=[vocab_size,4716],dtype=tf.float32,initializer = tf.ones_initializer())
    weights = tf.multiply(mask,weights[None,:,:])
    weights = tf.nn.softmax(weights, dim = 1)

    output = tf.multiply(weights, old_predictions)
    output = tf.reduce_sum(output, axis = 1)
    
    return {"predictions": output}

class SingleHiddenLayerEnsemble(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, old_predictions, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    # num_models = tf.shape(old_predictions)[1]
    old_predictions = old_predictions[:tf.shape(model_input)[0],:,:]
    hidden = slim.fully_connected(
        model_input, 512, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    weights = slim.fully_connected(
        hidden, vocab_size, activation_fn=tf.identity,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    weights = tf.nn.softmax(weights, dim = -1)

    output = tf.multiply(weights[:,:,None], old_predictions)
    output = tf.reduce_sum(output, axis = 1)
    
    return {"predictions": output}

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
