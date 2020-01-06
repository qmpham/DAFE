import tensorflow as tf
import numpy as np
from layers import common
from opennmt.utils import misc
from opennmt.utils.misc import shape_list

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
    outputs = self.outer(inner)
    self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(tf.reshape(outputs,[-1,tf.shape(outputs)[-1]])),axis=-1)))
    if not training:
      tf.print("#######")
      tf.print(self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), sep="|")
      tf.print("#######")
    return outputs

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
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v2, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v2, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v3(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v3, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v3, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v1(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.tanh,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v1, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v1, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(inner),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(inner),axis=-1)))

    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(inner)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(inner)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_Gate(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.sigmoid,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_Gate, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    #self.input_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.outer_transpose = False
    self.outer_use_bias = True
    self.outer_activation = activation
  
  def build(self, input_shape):
    super(Multi_domain_Gate, self).build(input_shape)
    scope_name = self.name_scope()
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    #inputs = self.input_norm(inputs)
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inputs, dom_outer_kernel, transpose_b=self.outer_transpose)
    
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    outputs = self.layer_norm(outputs)

    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    #if not training:
      #tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_gate_max_abs_pooling: ", 
      #          tf.reduce_max(tf.abs(outputs)), "ADAP_gate_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "ADAP_gate_avg_abs_pooling: ", tf.reduce_mean(tf.abs(outputs)), "domain: ", domain, "###", sep="|")
      
    #  tf.print("###", self.name_scope(), "ADAP_gate: ", outputs[0:2,tf.math.floordiv(tf.shape(outputs)[1],2),:], summarize=2048)

    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    
    outer_kernel = args_dict[self.outer_kernel.name]
    outer_bias = args_dict[self.outer_bias.name]
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inputs, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    outputs = self.layer_norm(outputs)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class DAFE(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               domain_numb=6,
               dropout=0.1,
               **kwargs):
    
    super(DAFE, self).__init__(**kwargs)
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.layer_norm = common.LayerNorm()

  def build(self, input_shape):
    super(DAFE, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    
  def call(self, inputs, domain, mask=None,  training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    outputs = tf.nn.bias_add(inputs, dom_inner_bias)    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])

    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    outputs = tf.nn.bias_add(inputs, dom_inner_bias)    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs