"""Define the self-attention encoder."""

import tensorflow as tf

from layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common
from layers.common import LayerNorm
from utils.utils_ import make_domain_mask
from layers.layers import Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, DAFE
class Multi_domain_SelfAttentionEncoder(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.mask = make_domain_mask(num_domains, num_domain_units=num_domain_units)
    self.multi_domain_layers = [
        Multi_domain_FeedForwardNetwork(num_domains*num_domain_units, num_units, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain_mask, training=training) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain_mask, training=training) + inputs
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain_mask, training=training) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain_mask, training=training) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m


class Multi_domain_SelfAttentionEncoder_v2(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v2, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [
        Multi_domain_FeedForwardNetwork_v2(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
                activity_regularizer=tf.compat.v1.keras.layers.ActivityRegularization(l1=1.0,l2=1.0))
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain) + inputs
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m


class Multi_domain_SelfAttentionEncoder_v3(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v3, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [
        DAFE(num_units, domain_numb=num_domains, name="DAFE_%d"%i)
        for i in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain)
      else:
        inputs = multi_domain_layer(inputs, domain)
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
