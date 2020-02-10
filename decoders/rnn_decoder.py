from opennmt.decoders.rnn_decoder import RNNDecoder
import tensorflow_addons as tfa
import tensorflow as tf
from layers import common

class AttentionalRNNDecoder(RNNDecoder):
  """A RNN decoder with attention."""

  def __init__(self,
               num_layers,
               num_units,
               bridge_class=None,
               attention_mechanism_class=None,
               cell_class=None,
               dropout=0.3,
               residual_connections=False,
               first_layer_attention=False,
               **kwargs):
    
    super(AttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge_class=bridge_class,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections,
        **kwargs)
    if attention_mechanism_class is None:
      attention_mechanism_class = tfa.seq2seq.LuongAttention
    self.attention_mechanism = attention_mechanism_class(self.cell.output_size)

    def _add_attention(cell):
      attention_layer = common.Dense(cell.output_size, use_bias=False, activation=tf.math.tanh)
      wrapper = tfa.seq2seq.AttentionWrapper(
          cell,
          self.attention_mechanism,
          attention_layer=attention_layer)
      return wrapper

    if first_layer_attention:
      self.cell.cells[0] = _add_attention(self.cell.cells[0])
    else:
      self.cell = _add_attention(self.cell)
    self.dropout = dropout
    self.first_layer_attention = first_layer_attention

  @property
  def support_alignment_history(self):
    return True

  def _get_initial_state(self, batch_size, dtype, initial_state=None):
    self.attention_mechanism.setup_memory(
        self.memory, memory_sequence_length=self.memory_sequence_length)
    decoder_state = self.cell.get_initial_state(batch_size=batch_size, dtype=dtype)
    if initial_state is not None:
      if self.first_layer_attention:
        cell_state = list(decoder_state)
        cell_state[0] = decoder_state[0].cell_state
        cell_state = self.bridge(initial_state, cell_state)
        cell_state[0] = decoder_state[0].clone(cell_state=cell_state[0])
        decoder_state = tuple(cell_state)
      else:
        cell_state = self.bridge(initial_state, decoder_state.cell_state)
        decoder_state = decoder_state.clone(cell_state=cell_state)
    return decoder_state

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    outputs, state = self.cell(inputs, state, training=training)
    outputs = common.dropout(outputs, self.dropout, training=training)
    if self.first_layer_attention:
      attention = state[0].alignments
    else:
      attention = state.alignments
    return outputs, state, attention