import tensorflow as tf
import numpy as np

from opennmt.layers import common
from opennmt.utils import misc


class Multi_domain_FeedForwardNetwork(tf.keras.layers.Layer):

  def __init__(self,
               inner_dim,
               output_dim,
               dropout=0.1,
               activation=tf.nn.relu,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork, self).__init__(**kwargs)
    self.inner = common.Dense(inner_dim, activation=activation)
    self.outer = common.Dense(output_dim)
    self.dropout = dropout
    self.layer_norm = common.LayerNorm()

  def call(self, inputs, mask, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_nor
    m(inputs)
    inner = self.inner(inputs)
    #print("inner: ", inner)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer(inner)