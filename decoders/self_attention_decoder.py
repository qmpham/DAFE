"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.decoders.decoder import Decoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from layers import common, transformer
from opennmt.layers.position import SinusoidalPositionEncoder
from layers.layers import Multi_domain_FeedForwardNetwork, Multi_domain_Gate_v2, Multi_domain_FeedForwardNetwork_v2, DAFE, Multi_domain_Gate, Multi_domain_FeedForwardNetwork_v3
from utils.utils_ import make_domain_mask
class Multi_domain_SelfAttentionDecoder(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.mask = make_domain_mask(num_domains, num_domain_units=num_domain_units)
    self.multi_domain_layers = [
        Multi_domain_FeedForwardNetwork(num_domains*num_domain_units, num_units, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
  def initialize(self, vocab_size=None, output_layer=None):
    
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain_mask) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain_mask) + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain_mask) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain_mask) + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v1(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v1, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient

  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Multi_ADAP_Dense(vocab_size, self.num_units, Multi_domain_FeedForwardNetwork_v2)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs, domain)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs, domain)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict, domain)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v2(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v2, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v3(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               multi_domain_adapter_class=DAFE,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v3, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="DAFE_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient

  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v5(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v5, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution=[1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Multi_ADAP_Dense_v1(vocab_size, self.num_units, Multi_domain_FeedForwardNetwork_v2)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs, domain)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs, domain)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict, domain)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v6(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v6, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
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
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v7(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v7, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
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
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v8(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v8, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
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
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      inputs = multi_domain_layer(g*inputs+(1-g)*tf.stop_gradient(inputs), domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
      inputs = multi_domain_layer.forward_fn(g*inputs + (1-g)*tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v9(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v9, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
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

    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient: 
        ADAP_input = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
      else:
        ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
      """
      if not training:
        tf.print("%s"%self.name_scope(), "forget_gate: ", f, "inputs gate:", i)
      """
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def _adv_run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      """
      if not training:
        tf.print("%s"%self.name_scope(), "forget_gate: ", f, "inputs gate:", i)
      """
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def adv_forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._adv_run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    #logits = self.output_layer(outputs)
    #logits = inputs[0]
    logits = outputs
    return logits, state, attention

  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
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
        ADAP_input = multi_domain_layer.forward_fn(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v0(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
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
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v0, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]

  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      inputs = multi_domain_layer(inputs, domain, mask=mask, training=training)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention