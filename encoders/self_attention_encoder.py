"""Define the self-attention encoder."""
import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

import tensorflow as tf

from layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common
from layers.common import LayerNorm, Multi_LayerNorm
from utils.utils_ import make_domain_mask
from layers.layers import Regulation_Gate, Multi_domain_classification_gate, Multi_domain_FeedForwardNetwork_v9, Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8, Multi_domain_FeedForwardNetwork_v7, Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, Multi_domain_FeedForwardNetwork_v3, DAFE, Multi_domain_Gate, Multi_domain_Gate_v2
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
    self.num_domains = num_domains
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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      inputs = layer(inputs, mask=mask, training=training)
      
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        inputs = adapt * self.ADAP_contribution[i] + inputs
      else:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        inputs = adapt * self.ADAP_contribution[i] + inputs
      """
      if internal_node_printing:
        tf.print("layers: ", i , "ADAP mean pooling: ", tf.reduce_mean(tf.abs(adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
      """
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
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
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

class Multi_domain_SelfAttentionEncoder_v0(Encoder):

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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v0, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
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
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      inputs = multi_domain_layer(inputs, domain, mask=mask, training=training)
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
      inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v1(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v1, self).__init__(**kwargs)
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
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
      if internal_node_printing:
        tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(tf.abs(g),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)

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
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v5(Encoder):
  
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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v5, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      inputs = multi_domain_layer(g * inputs + (1-g) * tf.stop_gradient(inputs), domain, mask=mask, training=training) + inputs
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
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      g = multi_domain_gate.forward_fn(inputs, domain, mask=mask, training=training)
      inputs = multi_domain_layer.forward_fn(g*inputs+(1-g)*tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m    

class Multi_domain_SelfAttentionEncoder_v4(Encoder):

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
               multi_domain_adapter_class=Multi_domain_Gate,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v4, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
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
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      g = multi_domain_gate.forward_fn(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v6(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               input_gate_regularization=False,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v6, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate")
    self.multi_domain_input_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate", output_regularization=input_gate_regularization)
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      #if not training:
      #  tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i)),sep="|")
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def adv_call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)      
      ADAP_input = tf.stop_gradient(multi_domain_layer(inputs, domain, mask=mask, training=training))
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      
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
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient: 
        ADAP_input = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
      else:
        ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v8(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               input_gate_regularization=False,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v8, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_input_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate_%d"%i, output_regularization=input_gate_regularization)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      f = multi_domain_forget_gate(tf.stop_gradient(inputs), tf.stop_gradient(ADAP_input), mask=mask, training=training)
      i_ = multi_domain_input_gate(tf.stop_gradient(inputs), tf.stop_gradient(ADAP_input), mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
      if not training:
        tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i_)),sep="|", output_stream=sys.stdout)
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def adv_call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs = layer(inputs, mask=mask, training=training)      
      ADAP_input = tf.stop_gradient(multi_domain_layer(inputs, domain, mask=mask, training=training))
      f = multi_domain_forget_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
      i_ = multi_domain_input_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
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
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate.forward_fn(inputs, mask=mask, training=training)
      i_ = multi_domain_input_gate.forward_fn(inputs, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v9(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Regulation_Gate,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v9, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
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
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v7(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v7, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate")
    self.multi_domain_input_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate")
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer in self.layers:
      inputs = layer(inputs, mask=mask, training=training)
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
    for layer in self.layers:
      inputs = layer.forward_fn(inputs, mask=mask, training=training)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v10(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v10, self).__init__(**kwargs)
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
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gate = transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)

    if self.ADAP_gate_stopping_gradient:
      g = self.multi_domain_gate(tf.stop_gradient(inputs), mask=mask, training=training)
    else:
      g = self.multi_domain_gate(inputs, mask=mask, training=training)

    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
      if internal_node_printing:
        tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(tf.abs(g),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)

    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v11(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               input_gate_regularization=False,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v11, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate")
                                    for i in range(num_layers)]
    self.multi_domain_input_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, 
                                    domain_numb=num_domains, name="ADAP_input_gate", output_regularization=input_gate_regularization)
                                    for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_forget_gate, multi_domain_input_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_forget_gates, self.multi_domain_input_gates):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      #if not training:
      #  tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i)),sep="|")
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
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient: 
        ADAP_input = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
      else:
        ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v12(Encoder):

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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               inner_layer_norm=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v12, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer(layer(tf.stop_gradient(inputs), mask=mask, training=training), domain, mask=mask, training=training)
        total_adapt.append(adapt)
        inputs = layer(inputs, mask=mask, training=training)                
      else:
        inputs = layer(inputs, mask=mask, training=training)
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
      
    total_adapt = tf.add_n(total_adapt)
    if internal_node_printing:
        tf.print("Encoder ADAP mean pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    outputs = self.layer_norm(inputs+total_adapt)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    #for layer in self.layers:
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer.forward_fn(layer(tf.stop_gradient(inputs), args_dict, mask=mask, training=training), args_dict, domain, mask=mask, training=training)
        total_adapt.append(adapt)
        inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)                
      else:
        inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
        adapt = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    outputs = self.layer_norm.forward_fn(inputs + tf.add_n(total_adapt), args_dict)
    return outputs, None, sequence_length

class Multi_domain_SelfAttentionEncoder_v15(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v15, self).__init__(**kwargs)
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
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gate = multi_domain_adapter_gate_class(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
      total_adapt.append(adapt)

    g = self.multi_domain_gate(inputs, domain, mask=mask, training=training)
    total_adapt = tf.add_n(total_adapt)
    if internal_node_printing:
      tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(g,-1)[0,:], "adapt_mean_abs_pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)  

    if self.ADAP_gate_stopping_gradient:
      print("stopping gradient at d_classifier in encoder")
      g = tf.stop_gradient(g)
    outputs = self.layer_norm(inputs * (1-g) + total_adapt * g)
    ##### z_k and ADAP_k(h) should align
    if training:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.norm(total_adapt*g,-1),-1),-1))
      tf.print("z_adap_agreement_loss: ", tf.reduce_mean(tf.reduce_sum(tf.norm(total_adapt*g,-1),-1),-1))
    return outputs, None, sequence_length

  def adv_call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
      total_adapt.append(adapt)

    g = self.multi_domain_gate(inputs, domain, mask=mask, training=training)
    total_adapt = tf.add_n(total_adapt)
    g = tf.stop_gradient(g)
    total_adapt = tf.stop_gradient(total_adapt)
    outputs = self.layer_norm(inputs * (1-g) + total_adapt * g)
    
    return outputs, None, sequence_length
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v16(Encoder):

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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v16, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_contribution[i]>0:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        inputs = adapt * self.ADAP_contribution[i] + inputs
      """
      if internal_node_printing:
        tf.print("layers: ", i , "ADAP mean pooling: ", tf.reduce_mean(tf.abs(adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
      """
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v17(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v17, self).__init__(**kwargs)
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
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
      total_adapt.append(adapt)

    g = multi_domain_gate(inputs, domain, mask=mask, training=training)
    total_adapt = tf.add_n(total_adapt)
    if internal_node_printing:
      tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(g,-1)[0,:], "adapt_mean_abs_pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)

    outputs = self.layer_norm(inputs * (1-g) + total_adapt * g)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v18(Encoder):

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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
               inner_layer_norm=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v18, self).__init__(**kwargs)
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
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_contribution[i]>0:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    if len(total_adapt)>0:
      total_adapt = tf.add_n(total_adapt)
    else:
      total_adapt = 0
    if internal_node_printing:
        tf.print("Encoder ADAP mean pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    outputs = self.layer_norm(inputs+total_adapt)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

















































