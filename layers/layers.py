import tensorflow as tf
import numpy as np
from layers import common
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
    inputs = self.layer_norm(inputs)
    inner = self.inner(inputs)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer(inner)

  def forward_fn(self, inputs, args_dict, mask, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    inner = self.inner.forward_fn(inputs, args_dict)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer.forward_fn(inner, args_dict)


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
    inputs = self.layer_norm(inputs)
    inner = self.inner(inputs)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer(inner)

  def forward_fn(self, inputs, args_dict, mask, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    inner = self.inner.forward_fn(inputs, args_dict)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer.forward_fn(inner, args_dict)


class Multi_domain_FeedForwardNetwork_v2(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v2, self).__init__(**kwargs)
    scope_name = self.name_scope()
    inner_weight = self.add_weight("%s_inner_weight"%scope_name, shape=[domain_numb, input_dim*inner_dim])
    inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[domain_numb, inner_dim])
    outer_weight = self.add_weight("%s_outer_weight"%scope_name, shape=[domain_numb, inner_dim*output_dim])
    outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[domain_numb, output_dim])
    self.inner = common.Multi_Dense(inner_dim, inner_weight, inner_bias, activation=activation)
    self.outer = common.Multi_Dense(output_dim, outer_weight, outer_bias)
    self.dropout = dropout
    self.layer_norm = common.LayerNorm()

  def call(self, inputs, domain, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    inner = self.inner.call_indomain(domain)(inputs)    
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer(domain)(inner)

  def forward_fn(self, inputs, args_dict, domain, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    inner = self.inner.forward_fn_indomain(domain)(inputs, args_dict)    
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer.forward_fn(domain)(inner, args_dict)

