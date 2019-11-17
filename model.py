# -*- coding: utf-8 -*-

"""Standard sequence-to-sequence model."""

import six

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import constants
from opennmt import inputters
from opennmt import layers
from opennmt.models.sequence_to_sequence import SequenceToSequence
from opennmt.data import noise
from opennmt.data import text
from opennmt.data import vocab
from opennmt.layers import reducer
from opennmt.models import model
from opennmt.utils import decoding
from opennmt.utils import losses
from opennmt.utils.misc import print_bytes, format_translation_output, merge_dict, shape_list
from opennmt.decoders import decoder as decoder_util
from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel, SequenceToSequence, SequenceToSequenceInputter, replace_unknown_target, _add_noise
from utils.my_inputter import My_inputter, Multi_domain_SequenceToSequenceInputter
from utils.utils_ import make_domain_mask, masking
from opennmt.layers import common
from opennmt.utils.losses import _softmax_cross_entropy
class Multi_domain_SequenceToSequence(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings

  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)

    return outputs, predictions

  def forward_fn(self, features, args_dict, labels=None, training=None, step=None):
    # Encode the source.
    training=True
    assert labels!=None
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter.forward_fn(features, args_dict, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.forward_fn(
        [source_inputs, features["domain"]], args_dict, sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target_forward_fn(
          labels,
          args_dict,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    return tf.reduce_sum(encoder_outputs)#outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"]],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs

  def _decode_target_forward_fn(self,
                     labels,
                     args_dict,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter.forward_fn(labels, args_dict, training=training)
    input_fn = lambda ids: [self.labels_inputter.forward_fn({"ids": ids}, args_dict, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
        
    logits, _, attention = self.decoder.forward_fn(
        [target_inputs, labels["domain"]],
        args_dict,
        self.labels_inputter.get_length(labels),
        initial_state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder.forward_fn(
          noisy_inputs,
          args_dict,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs
 
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"]],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

class LDR_SequenceToSequence(model.SequenceGenerator):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(LDR_SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings

  def auto_config(self, num_replicas=1):
    config = super(LDR_SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(LDR_SequenceToSequence, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(LDR_SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        source_inputs, sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        target_inputs,
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    print("label_smoothing: ",params.get("label_smoothing", 0.0))
    print("average_loss_in_time: ", params.get("average_loss_in_time", False))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(LDR_SequenceToSequence, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)

class Masked_LM(model.Model):
  def __init__(self,
               source_inputter,
               encoder):

    super(Masked_LM, self).__init__(source_inputter)
    self.encoder = encoder

  def build(self, input_shape):
    super(Masked_LM, self).build(input_shape)
    vocab_size = self.examples_inputter.vocabulary_size
    output_layer = None
    if self.reuse_embedding:
      output_layer = layers.Dense(
          vocab_size,
          weight=self.examples_inputter.embedding,
          transpose=True)
    self.decoder.initialize(vocab_size=vocab_size, output_layer=output_layer)

  def call(self, features, training=None, step=None):
    
    ids, length = features["ids"], features["length"]
    #ids = masking(ids, noise_percentage=0.15)
    logits, _ = self.encoder(ids, length, training=training)
    outputs = dict(logits=logits)
    
    return outputs

  def compute_loss(self, outputs, features, training=True):
    ids_out = None
    length = None
    logits = outputs["logits"]
    labels = features["ids"]
    sequence_length = features["length"]
    label_smoothing = 0.1
    average_in_time = True
    batch_size = tf.shape(logits)[0]
    max_time = tf.shape(logits)[1]

    cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, training)
    weights = tf.sequence_mask(
        sequence_length, maxlen=max_time, dtype=cross_entropy.dtype)
    loss = tf.reduce_sum(cross_entropy * weights)
    loss_token_normalizer = tf.reduce_sum(weights)

    if average_in_time or not training:
      loss_normalizer = loss_token_normalizer
    else:
      loss_normalizer = tf.cast(batch_size, loss.dtype)

    return loss, loss_normalizer, loss_token_normalizer
  