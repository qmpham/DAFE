"""Define the self-attention encoder."""

import tensorflow as tf

from layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common
from layers.common import LayerNorm
from utils.utils_ import make_domain_mask
from layers.layers import Multi_domain_FeedForwardNetwork
class Multi_domain_SelfAttentionEncoder(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
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
      inputs = multi_domain_layer(inputs, domain_mask) + inputs
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
      inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain_mask) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
