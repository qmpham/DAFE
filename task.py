import argparse
import logging
import yaml
import tensorflow as tf
import tensorflow_addons as tfa
import sys
import opennmt as onmt
import io
import os
import utils
from opennmt import START_OF_SENTENCE_ID
from opennmt import END_OF_SENTENCE_ID
from opennmt.utils.misc import print_bytes
from opennmt.data import dataset as dataset_util
from opennmt.optimizers import utils as optimizer_util
tf.get_logger().setLevel(logging.INFO)
from utils.my_inputter import My_inputter, LDR_inputter
from opennmt.models.sequence_to_sequence import SequenceToSequence
from model import Multi_domain_SequenceToSequence, LDR_SequenceToSequence
from encoders.self_attention_encoder import Multi_domain_SelfAttentionEncoder
from decoders.self_attention_decoder import Multi_domain_SelfAttentionDecoder
import numpy as np
from utils.dataprocess import merge_map_fn, create_trainining_dataset_v1, create_multi_domain_meta_trainining_dataset_v2, create_meta_trainining_dataset, create_trainining_dataset, create_multi_domain_meta_trainining_dataset, create_trainining_dataset_v2, create_multi_domain_meta_trainining_dataset_v1
from opennmt.utils import BLEUScorer
from opennmt.inputters.text_inputter import WordEmbedder
from utils.utils_ import variance_scaling_initialier, MultiBLEUScorer, var_spec
from layers.layers import Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, DAFE
from utils.utils_ import average_checkpoints, load_and_update_if_needed_from_ckpt
from utils.dataprocess import count_lines

def update(v,g,lr=1.0):
  if isinstance(g, tf.IndexedSlices):
    return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
  else:
    return v-lr*g

def translate(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              checkpoint_path=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"]], source_length)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","residualv15","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def translate_with_tag_file(source_file,
              tag_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              output_file,
              length_penalty,
              checkpoint_path=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=1,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("with domain tag: %s"%tag_file)
  checkpoint.restore(checkpoint_path)
  domain = 0
  dataset = model.examples_inputter.make_inference_dataset(source_file, 1, domain)
  tag_dataset = tf.data.TextLineDataset(tag_file)
  tag_dataset = tag_dataset.batch(1)
  iterator = iter(dataset)
  tag_iterator = iter(tag_dataset)
  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    tag = next(tag_iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    tag_inputs = tf.strings.to_number(tag, out_type=tf.int64)
    if experiment in ["residual","residualv15","DC1","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, tag_inputs], source_length)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","residualv15","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), tag_inputs[0]), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=tag_inputs[0])
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def debug(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100,
          picking_prob=None): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)

  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    num_examples = tf.reduce_sum(meta_train_target["length"])
    domain = meta_train_source["domain"][0]
    tf.print("domain:",domain)
    #tf.print("token_numb:____", num_examples, "domain:____", meta_train_source["domain"][0])
    reported_loss = 0
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples, domain

  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      print(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target)
      per_replica_loss, per_replica_num_word_examples, per_replica_domain = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None) 
      domain = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_domain, None)
    return loss, num_word_examples, domain
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  step=0
  count = np.array([0] * len(source_file))
  with _summary_writer.as_default():
    while True:
      #####Training batch
      _, num_examples, domain = next(meta_train_data_flow) 
      domain = domain.numpy()
      #print(domain)
      count += np.array([1 if i==int(domain) else 0 for i in range(len(source_file))])
      step +=1
      if step % 100 == 0:
        print([c/sum(count) for c in count])
      #_loss.append(loss)
      #print("number examples per replica: ", num_examples)
      #print(next(meta_train_flow))

def meta_train_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    meta_train_gradient_accumulator = optimizer_util.GradientAccumulator()  
    meta_test_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_meta_train_gradients(source, target):
    print("source: ", source)
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
        variables.append(variable)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    meta_train_gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_meta_test_gradients(source, target):
    print("source: ", source)
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
        variables.append(variable)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    meta_test_gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_meta_train_gradients():
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
        variables.append(variable)
    print("var numb: ", len(variables))
    grads_and_vars = []
    
    for gradient, variable in zip(meta_train_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      #if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(meta_train_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    meta_train_gradient_accumulator.reset()

  def _apply_meta_test_gradients():
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
        variables.append(variable)
    print("var numb: ", len(variables))
    grads_and_vars = []
    
    for gradient, variable in zip(meta_test_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      #if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(meta_test_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    meta_test_gradient_accumulator.reset()
 
  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_meta_train_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, _ = strategy.experimental_run_v2(
          _accumulate_meta_test_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
    return loss

  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_iteration(next_fn):    
    with strategy.scope():
      return next_fn()
  
  @tf.function
  def _meta_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_meta_train_gradients)

  @tf.function
  def _meta_test_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_meta_test_gradients)

  def _set_weight(v, w):
    v.assign(w)

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  meta_test_data_flow = iter(_meta_test_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      #print("number_examples_per_replica: ", num_examples)
      _loss.append(loss)  
      #snapshots = [v.value() for v in model.trainable_variables]
      _meta_train_step()
      #####Testing batch
      loss = next(meta_test_data_flow)
      #weight_reset(snapshots)
      _meta_test_step()
      ####      
      step = optimizer.iterations.numpy()//2
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0 and optimizer.iterations.numpy()%2==0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0 and optimizer.iterations.numpy()%2==0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables()
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    tf.print("meta_train_source: ",meta_train_source, output_stream=sys.stderr)
    tf.print("meta_train_target: ",meta_train_target, output_stream=sys.stderr)
    tf.print("meta_test_source: ",meta_test_source, output_stream=sys.stderr)
    tf.print("meta_test_target: ",meta_test_target, output_stream=sys.stderr)
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    training_loss = loss[0] / loss[1]
    reported_loss = loss[0] / loss[2]
    variables = model.trainable_variables       
    args_dict = dict()
    for v in variables:
      args_dict.update({v.name:v})
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = tf.gradients(training_loss, variables)
    ##### Inner adaptation
    def update(v,g,lr=1.0):
      if "embedding" in v.name:
        # print("embedding gradient's values: __________", g.values)
        # print("embedding gradient's indices: _________", g.indices)
        print(v)
        print(g)
        return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
      else:
        return v-lr*g
    if config.get("stopping_gradient",True):
      print("apply stopping_gradient")
      for g, v in zip(gradients, variables):      
        args_dict.update({v.name: v-g})
    else:
      print("passing gradient")
      for g, v in zip(gradients, variables):
        args_dict.update({v.name: update(v,g)})
    #### Meta_loss:
    outputs, _ = model.forward_fn(meta_test_source,
        args_dict,
        labels=meta_test_target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, meta_test_target, training=True)
    meta_training_loss = loss[0] / loss[1]
    meta_reported_loss = loss[0] / loss[2]
    meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
    gradients = optimizer.get_gradients(meta_training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(meta_test_target["length"])[0]
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      _loss.append(loss)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def elastic_finetuning(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  star_vars = []

  def build_model(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
  
  @dataset_util.function_on_next(train_dataset)
  def _build_model(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          build_model, args=(per_replica_source, per_replica_target))

  @tf.function
  def star_vars_init():
    variables = model.trainable_variables
    with tf.init_scope():
      for var in variables:
        value=var.numpy()
        star_vars.append(tf.constant(value))

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    lambda_ = config.get("lambda", 0.001)
    print("elastic weights: ", lambda_)
    for i in range(len(variables)):
      training_loss += tf.reduce_sum(tf.square(variables[i] - star_vars[i])) * lambda_
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  first_run = iter(_build_model())
  next(first_run)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()     
  ###
  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        model_key="model")

  ### assign value to star_vars
  star_vars_init()

  ###
  train_data_flow = iter(_train_forward())
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        if config.get("unsupervised_clustering",False):
          tag_files = config.get("tag_files")
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          if config.get("unsupervised_clustering",False):
            score = translate_with_tag_file(src, tag_files[i], ref, model, checkpoint_manager, checkpoint, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          else:
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def finetuning(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,          
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  checkpoint_path = config.get("checkpoint_path",None)
  #####
  if checkpoint_path is None:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = []
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        print(v.name)
        variables.append(v)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = []
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _finetuning_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  finetuning_data_flow = iter(_finetuning_forward())
  
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(finetuning_data_flow)
      _step()     
      _loss.append(loss)
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v7(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      shared_gradients = []
      adap_gradients = []
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      variables = adap_variables + shared_variables
      adap_variables_name = [v.name for v in adap_variables]
      shared_variables_name = [v.name for v in shared_variables]
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 
      var_spec(variables)
      var_spec(shared_variables)
      var_spec(adap_variables)
      for g,v in zip(gradients, variables):
        if v.name in shared_variables_name:
          shared_gradients.append(g)
        elif v.name in adap_variables_name:
          adap_gradients.append(g)

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(shared_gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(shared_gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### update adap parameters first
      outputs, _ = model(
          meta_test_source,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_test_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_test_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      adap_gradients = tape.gradient(training_loss, adap_variables)
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      shared_gradients = tape.gradient(meta_training_loss, shared_variables)
      gradients = adap_gradients + shared_gradients
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    adap_variables = []
    shared_variables = []
    for v in variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        adap_variables.append(v)
      else:
        shared_variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, adap_variables+shared_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v3(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    """
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    """
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      ##### Inner adaptation
      training_loss = model.regularize_loss(training_loss, variables=adap_variables)
      gradients = tape.gradient(training_loss, adap_variables)    
      meta_train_lr = config.get("meta_train_lr", 0.1)
      print("meta_train_lr: ", meta_train_lr)

      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, adap_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, adap_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables      
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v5(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    """
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    """
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      ##### Inner adaptation
      training_loss = model.regularize_loss(training_loss, variables=shared_variables)
      gradients = tape.gradient(training_loss, shared_variables)    
      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v6(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      training_loss = model.regularize_loss(training_loss, variables=shared_variables)
      gradients = tape.gradient(training_loss, shared_variables)    
      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
        #### update adap parameters first
      outputs, _ = model(
          meta_test_source,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      training_loss = loss[0] / loss[1]
      gradients = tape.gradient(training_loss, adap_variables)
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=shared_variables)
      gradients.extend(tape.gradient(meta_training_loss, shared_variables))
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    adap_variables = []
    shared_variables = []
    for v in variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        adap_variables.append(v)
      else:
        shared_variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, adap_variables+shared_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def train(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    for var in variables:
      print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        if config.get("unsupervised_clustering",False):
          tag_files = config.get("tag_files")
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          if config.get("unsupervised_clustering",False):
            score = translate_with_tag_file(src, tag_files[i], ref, model, checkpoint_manager, checkpoint, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          else:
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break
  
def train_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=5000000,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  length_bucket_width = config.get("length_bucket_width",1)
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset_v2(strategy, model, domain, source_file, target_file, 
                                              batch_train_size, batch_type, shuffle_buffer_size, maximum_length, 
                                              length_bucket_width, multi_domain=True)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("token_numb:____", num_examples, "domain:____", source["domain"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break
    
def meta_train_v8(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
    #tf.print("meta_train_domain", meta_train_source["domain"][0], "meta_test_domain: ", meta_test_source["domain"][0], sep="|")
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v15(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
    #tf.print("meta_train_domain", meta_train_source["domain"][0], "meta_test_domain: ", meta_test_source["domain"][0], sep="|")
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      #gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v10(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset_v1(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v11(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset_v1(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 
      
      meta_train_lr = learning_rate(optimizer.iterations)      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v9(config,
          optimizer_1,
          optimizer_2,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer_1.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables    
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      meta_train_lr = learning_rate(optimizer_1.iterations)
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer_1.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables    
    grads_and_vars = []
    shared_grads_and_vars = []
    adap_grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      if "ADAP" in variable.name:
        adap_grads_and_vars.append((scaled_gradient, variable))
      else:
        shared_grads_and_vars.append((scaled_gradient, variable))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer_1.apply_gradients(shared_grads_and_vars)
    optimizer_2.apply_gradients(adap_grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer_1.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def model_inspect(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=6,
          batch_size = 1,
          batch_type = "examples",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = batch_size
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))
  print("batch_size", batch_size)


  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    regularization_losses = model.losses #model.regularization_losses
    print(regularization_losses)
    loss = loss + tf.add_n([loss for loss in regularization_losses if model.name_scope() in loss.name])
    variables = model.trainable_variables
    print("trainable_variables: ")
    for v in variables:
      print(v.name)
      print(v.shape)
    return loss
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      
    return loss
  
  # Runs the training loop.
  train_data_flow = iter(_train_forward())
  for i in range(6):
    _ = next(train_data_flow)

def src_wemb_pretrain(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

def train_v3(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset_v1(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def train_v8(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):     
    ##### Inner adaptation
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    training_loss = loss[0] / loss[1]
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = meta_train_source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = meta_train_source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables       
    gradients = tf.gradients(training_loss, variables)  
    gradient_accumulator(gradients)
    
    outputs, _ = model(meta_test_source,
        labels=meta_test_target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, meta_test_target, training=True)
    meta_training_loss = loss[0] / loss[1]
    gradients = tf.gradients(meta_training_loss, variables)
    gradient_accumulator(gradients)
    num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
  
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v12(config,
          inner_optimizer,
          outer_optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_datasets = create_multi_domain_meta_trainining_dataset_v2(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=outer_optimizer)
    _outer_gradient_accumulator = optimizer_util.GradientAccumulator()  
    _inner_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_meta_train_gradients(source, target):
    #print("source: ", source)
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=outer_optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = outer_optimizer.get_gradients(training_loss, variables)
    _outer_gradient_accumulator(gradients)
    _inner_gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("domain:",source["domain"][0])
    return reported_loss, num_examples

  def _apply_inner_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(_inner_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(_inner_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    _inner_gradient_accumulator.reset()

  def _apply_outer_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(_outer_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(_outer_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    outer_optimizer.apply_gradients(grads_and_vars)
    _outer_gradient_accumulator.reset()

  @tf.function
  def _inner_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_inner_gradients)
  @tf.function
  def _outer_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_outer_gradients)
  def _set_weight(v, w):
    v.assign(w)

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  meta_train_data_flows = []
  for meta_train_dataset in meta_train_datasets:
    @dataset_util.function_on_next(meta_train_dataset)
    def _meta_train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_meta_train_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)    
      return loss, num_examples

    meta_train_data_flow = iter(_meta_train_forward())
    meta_train_data_flows.append(meta_train_data_flow)

  # Runs the training loop.
  import time
  start = time.time()  
  #print("meta_train_data_flows: ", meta_train_data_flows)
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = [[]] * len(meta_train_data_flows)
  _num_word_examples = []
  inner_loop_numb = [int(2)] * len(meta_train_data_flows)
  with _summary_writer.as_default():
    while True:  
      ##save current value of variables
      snapshots = [v.value() for v in model.trainable_variables]    
      domain = np.random.choice(len(meta_train_data_flows),1,p=picking_prob)[0]      
      ##inner loop
      for _ in range(inner_loop_numb[domain]):
        loss, num_word_examples = next(meta_train_data_flows[domain])  
        _loss[domain].append(loss)  
        _num_word_examples.append(num_word_examples)
        _inner_train_step()
      ##outer loop
      weight_reset(snapshots)
      _outer_train_step()
      ####      
      step = outer_optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean([np.mean(losses) for losses in _loss]), np.sum(_num_word_examples), elapsed)
        _loss = [[]] * len(meta_train_data_flows)
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v13(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
    #tf.print("meta_train_domain", meta_train_source["domain"][0], "meta_test_domain: ", meta_test_source["domain"][0], sep="|")
    meta_train_source["domain"] = tf.tile(tf.expand_dims(meta_test_source["domain"][0],0), meta_train_source["domain"].shape)
    meta_train_target["domain"] = tf.tile(tf.expand_dims(meta_test_target["domain"][0],0), meta_train_target["domain"].shape)
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      #gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def train_v12(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=10000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_datasets = create_multi_domain_meta_trainining_dataset_v2(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_train_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("domain:",source["domain"][0])
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    #tf.print("accumulated_gradients: ",gradient_accumulator.step)
    gradient_accumulator.reset()

  @tf.function
  def train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  train_data_flows = []
  for train_dataset in train_datasets:
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_train_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)    
      return loss, num_examples

    train_data_flow = iter(_train_forward())
    train_data_flows.append(train_data_flow)

  # Runs the training loop.
  import time
  start = time.time()  
  #datasets_size = [count_lines(src) for src in source_file]
  #picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = [0.0] * len(train_data_flows)
  _num_word_examples = []
  step = 0
  importance_recalculate = config.get("importance_recalculate", 2000)
  save_stats = config.get("save_stats", 5000)
  domain_num = len(train_data_flows)
  # warmup_steps = config.get("warmup_steps",4000)
  # step_duration = config.get("step_duration",16)
  # prefinetuning_steps = config.get("prefinetuning_steps",200000)
  stats_path = os.path.join(config["model_dir"],"stats")
  if os.path.exists(stats_path):
    print("load %s"%stats_path)
    stats = np.load(stats_path)
  else:
    stats = {"consecutive_eval_drops": [0] * len(train_data_flows),
          "last_bleu_scores": [0] * len(train_data_flows),
          "last_training_loss": [20.0] * len(train_data_flows),
          "overfitting": [False] * len(train_data_flows),
          "consecutive_eval_drops:": [0] * len(train_data_flows),
          "importances": [1.0] * len(train_data_flows)}
  
  current_bleu_scores = [0] * domain_num
  current_training_loss = [0.0] * domain_num
  count = [1.0] * domain_num
  count_ = [1.0] * domain_num
  with _summary_writer.as_default():
    while True: 
      picking_prob = [importance/sum(stats["importances"]) for importance in stats["importances"]]
      domain = np.random.choice(domain_num,1,p=picking_prob)[0] 
      loss, num_word_examples = next(train_data_flows[domain])
      loss = loss.numpy()  
      _loss[domain] += loss
      count[domain] += 1
      current_training_loss[domain] += loss
      count_[domain] += 1
      _num_word_examples.append(num_word_examples)
      train_step()
      print("current_training_loss:",current_training_loss)
      ####      
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %s; num_word_examples = %d; after %f seconds; Importance = %s",
            step, learning_rate(step), " ".join([str(_loss[i]/count[i]) for i in range(len(_loss))]), np.sum(_num_word_examples), elapsed, " ".join([str(p) for p in picking_prob]))
        _loss = [0.0] * domain_num
        count = [1.0] * domain_num
        _num_word_examples = []
        start = time.time()

      if step % importance_recalculate:        
        current_training_loss = [current_training_loss[i]/count_[i] for i in range(domain_num)]
        print("last_training_loss:",stats["last_training_loss"])
        print("current_training_loss:",current_training_loss)
        for i in range(domain_num):
          if stats["last_training_loss"][i] < current_training_loss[i]:
            stats["importances"][i] = stats["importances"][i] * 2
          stats["last_training_loss"][i] = current_training_loss[i]
          current_training_loss[i] = 0.0
          count_[i] = 1.0

      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      
      if step % save_stats == 0:
        np.savez(os.path.join(config["model_dir"],"stats"), **stats)
      
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          current_bleu_scores[i] = score

        for i in range(domain_num):
          if stats["last_bleu_scores"][i] > current_bleu_scores[i]:
            stats["consecutive_eval_drops"][i] +=1
          else:
            stats["consecutive_eval_drops"][i] = 0
          
          if stats["consecutive_eval_drops"][i] > 2:
            stats["overfitting"][i] = True

          if stats["overfitting"][i]:
            stats["importances"][i] = stats["importances"][i] / 2
          
      if step > train_steps:
        break

def train_v13(config,
          optimizer, 
          adv_optimizer,         
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  domain_num = len(domain)
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  
  adv_train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"),picking_prob=None)

  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    gate_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    
    if config.get("input_gate_regularizing", False):
      regularization_losses = model.losses
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "input_gate" in loss_.name:
          output_activity_regularization_losses.append(loss_)
      print(output_activity_regularization_losses)
      training_loss -= tf.add_n(output_activity_regularization_losses) * config.get("input_gate_regularization_scale", 0.0001)

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
  
  def is_ADAP_learning_variable(name):

    yes = False

    for i in range(6):
      if "ADAP_%d"%i in name:
        yes= True
        break
    return yes

  def _accumulate_adv_gradients(source, target):
    adv_domain = tf.reshape(tf.math.floormod(source["domain"][0] + \
      tf.cast(tf.reshape(tf.random.categorical(tf.expand_dims(tf.math.log([1.0/(domain_num-1)]*domain_num),0),1),[]) + 1,tf.int32), domain_num), [])
    #tf.print("domain:", source["domain"][0],"adv_domain:", adv_domain, sep="|")
    source["domain"] = tf.tile(tf.expand_dims(adv_domain,0), tf.shape(source["domain"]))
    target["domain"] = tf.tile(tf.expand_dims(adv_domain,0), tf.shape(target["domain"]))
    outputs, _ = model.adv_call(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True) 
    training_loss = loss
    reported_loss = loss

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss * config.get("adv_loss_weight", 0.01), loss
    if config.get("input_gate_regularizing",False):
      regularization_losses = model.losses
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "input_gate" in loss_.name:
          output_activity_regularization_losses.append(loss_)
      print(output_activity_regularization_losses)
      training_loss += tf.add_n(output_activity_regularization_losses) * config.get("input_gate_regularization_scale", 0.0001)
    variables = [var for var in model.trainable_variables if not is_ADAP_learning_variable(var.name)]
    print("var numb: ", len(variables))
    gradients = adv_optimizer.get_gradients(training_loss, variables)
    gate_gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()

  def _apply_adv_gradients():
    variables = [var for var in model.trainable_variables if not is_ADAP_learning_variable(var.name)]
    grads_and_vars = []
    for gradient, variable in zip(gate_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gate_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    adv_optimizer.apply_gradients(grads_and_vars)
    gate_gradient_accumulator.reset()
  
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @dataset_util.function_on_next(adv_train_dataset)
  def _adv_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_adv_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _adv_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_adv_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  train_adv_data_flow = iter(_adv_train_forward())  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _adv_loss = []
  _number_examples = []    
  step = 0
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      
      if step > 30000:
        adv_loss, _ = next(train_adv_data_flow)
        _adv_step()
        _adv_loss.append(adv_loss)
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        if len(_adv_loss)!=0:
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Adv_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_adv_loss), np.sum(_number_examples), elapsed)
        else:
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _adv_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def domain_classification_on_top_encoder(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    logits = model.classification_on_top_encoder(source, training=True)
    training_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], logits)    
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])    
    return tf.reduce_mean(training_loss), num_examples

  def _apply_gradients():
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []      

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          domain_predict(src, model, checkpoint_path, checkpoint, i, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
      tf.summary.flush()
      if step > train_steps:
        break

def domain_predict(source_file,
              model,
              checkpoint_path,
              checkpoint,
              domain,
              length_penalty,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=5):
  
  # Create the inference dataset.
  checkpoint.restore(checkpoint_path)
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.

  @tf.function
  def predict_next():    
    source = next(iterator)  
    e, logits = model.classification_on_top_encoder(source, training=False)
    return tf.argmax(logits,-1)

  # Iterates on the dataset.
  
  predicted_domain = []
  
  while True:    
    try:
      predictions = predict_next()
      for d in predictions.numpy():          
        predicted_domain.append(d)
    except tf.errors.OutOfRangeError:
      break
  true_domain = [domain] * len(predicted_domain)
  from sklearn.metrics import classification_report
  from sklearn.metrics import accuracy_score
  print(classification_report(true_domain, predicted_domain))
  return accuracy_score(true_domain, predicted_domain)

def sentence_encode(source_file,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              experiment="ldr",
              batch_size=10):
  
  # Create the inference dataset.
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", checkpoint_manager.latest_checkpoint)
  print("In domain %d"%domain)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.

  @tf.function
  def encode_next():    
    source = next(iterator)  
    emb = model.sentence_encode(source, training=False)
    return emb

  # Iterates on the dataset.
  
  print("output file: ", output_file)
  src_sentence_embedding_list = []  
  maxcount = 1000000
  count = 0
  index = 0
  while True:    
    try:
      src_sentence_embedding_ = encode_next()
      src_sentence_embedding__ = src_sentence_embedding_.numpy()      
      src_sentence_embedding_list.append(src_sentence_embedding__)
      count += src_sentence_embedding__.shape[0]
      if count > maxcount:
        src_sentences = np.concatenate(src_sentence_embedding_list, axis=0)
        np.savez(output_file+str(index),sentence_embeddings=src_sentences)
        count = 0
        src_sentence_embedding_list = []
        index +=1
    except tf.errors.OutOfRangeError:
      break
  if len(src_sentence_embedding_list)>0:
    src_sentences = np.concatenate(src_sentence_embedding_list, axis=0)
    np.savez(output_file+str(index),sentence_embeddings=src_sentences)

def experimental_translate(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              encoder_domain,
              decoder_domain,
              output_file,
              length_penalty,
              checkpoint_path=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("encoder_domain: %d"%encoder_domain)
  print("decoder_domain: %s"%decoder_domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, encoder_domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","residualv16","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"]], source_length)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv15","residualv16","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), decoder_domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=decoder_domain)
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def visualize(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=None,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  
  batch_train_size = config["batch_train_size"]  
  batch_type = config.get("batch_type","tokens")
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), single_pass=True,
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations,
        internal_node_printing=True)
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
  
  # Runs the training loop.
  train_data_flow = iter(_train_forward())  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("Visualizing gating value")
  while True:  
    try:    
      next(train_data_flow)    
      tf.summary.flush()
    except StopIteration:
      break

def train_wdc(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    non_adv_gradient_accumulator = optimizer_util.GradientAccumulator()  
    adv_gradient_accumulator = optimizer_util.GradientAccumulator()
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    print(outputs)
    classification_logits_r = outputs["classification_logits_r"]
    classification_logits_s = outputs["classification_logits_s"]
    encoder_classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], classification_logits_r))
    probs = tf.nn.softmax(classification_logits_s, axis=1)
    prediction_probs = tf.map_fn(lambda x: x[0][x[1]], (probs, source["domain"]), dtype=tf.float32)
    adv_loss_1 = - tf.reduce_mean(tf.math.log(prediction_probs)) #tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], classification_logits_s))
    adv_loss_2 = - tf.reduce_mean(probs * tf.math.log(probs)) #- tf.reduce_mean(prediction_probs * tf.math.log(prediction_probs)) #- tf.reduce_mean(probs * tf.math.log(probs))
    #decoder_classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], outputs["state"])
    loss = model.compute_loss(outputs, target, training=True)  
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    total_loss = training_loss - adv_loss_2 * 0.2 + encoder_classification_loss 
    non_adv_vars = [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" not in v.name] + \
                    [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" in v.name and ("v_a" in v.name or "W_a" in v.name)]
    adv_vars = [v for v in model.trainable_variables if "ADV_on_top_encoder_domain_classification" in v.name and not ("v_a" in v.name or "W_a" in v.name)] 
    #####
    reported_loss = training_loss
    print("var numb: ", len(non_adv_vars))
    for v in non_adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(total_loss, non_adv_vars)
    non_adv_gradient_accumulator(gradients)
    #####
    print("adv_var_numb: ", len(adv_vars))
    for v in adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(adv_loss_1, adv_vars)
    adv_gradient_accumulator(gradients)
    #####
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, adv_loss_1, adv_loss_2, encoder_classification_loss, num_examples

  def _apply_gradients():
    grads_and_vars = []
    ####
    non_adv_vars = [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" not in v.name] + \
                    [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" in v.name and ("v_a" in v.name or "W_a" in v.name)]
    for gradient, variable in zip(non_adv_gradient_accumulator.gradients, non_adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(non_adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    #####
    adv_vars = [v for v in model.trainable_variables if "ADV_on_top_encoder_domain_classification" in v.name and not ("v_a" in v.name or "W_a" in v.name)] 
    for gradient, variable in zip(adv_gradient_accumulator.gradients, adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    non_adv_gradient_accumulator.reset()
    adv_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_adv_loss_1, per_replica_adv_loss_2, per_replica_encoder_classification_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)    
      adv_loss_1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_adv_loss_1, None) 
      adv_loss_2 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_adv_loss_2, None) 
      encoder_classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_encoder_classification_loss, None)   
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, adv_loss_1, adv_loss_2, encoder_classification_loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _adv_loss_1 = [] 
  _adv_loss_2 = []
  _encoder_classification_loss = []
  _number_examples = []

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, adv_loss_1, adv_loss_2, encoder_classification_loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _adv_loss_1.append(adv_loss_1)
        _adv_loss_2.append(adv_loss_2)
        _encoder_classification_loss.append(encoder_classification_loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Adv_loss_1 = %f, Adv_loss_2 = %f, Encoder_classification_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_adv_loss_1), np.mean(_adv_loss_2), np.mean(_encoder_classification_loss), np.sum(_number_examples), elapsed)
        _loss = []  
        _adv_loss_1 = [] 
        _adv_loss_2 = []
        _encoder_classification_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def train_ldr(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  generic_dataset = create_trainining_dataset(strategy, model, config["generic_domain"], config["generic_source_file"], config["generic_target_file"], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(generic_dataset)
  def _generic_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  generic_data_flow = iter(_generic_train_forward())
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()    

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _, _ = next(generic_data_flow)
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break 

def train_denny_britz(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    non_adv_gradient_accumulator = optimizer_util.GradientAccumulator()  
    adv_gradient_accumulator = optimizer_util.GradientAccumulator()
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    domain_classification_logits = outputs["domain_classification_logits"]
    encoder_classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], domain_classification_logits))
    loss = model.compute_loss(outputs, target, training=True)  
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config["adv_training"]:
      print("adv_training")
      total_loss = training_loss - encoder_classification_loss
    else:
      total_loss = training_loss + encoder_classification_loss
    non_adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" not in v.name]
    adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" in v.name] 
    #####
    reported_loss = training_loss
    print("var numb: ", len(non_adv_vars))
    for v in non_adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(total_loss, non_adv_vars)
    non_adv_gradient_accumulator(gradients)
    #####
    print("adv_var_numb: ", len(adv_vars))
    for v in adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(encoder_classification_loss, adv_vars)
    adv_gradient_accumulator(gradients)
    #####
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, encoder_classification_loss, num_examples

  def _apply_gradients():
    non_adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" not in v.name]
    grads_and_vars = []
    for gradient, variable in zip(non_adv_gradient_accumulator.gradients, non_adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(non_adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    non_adv_gradient_accumulator.reset()

  def _apply_adv_gradients():
    adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" in v.name]  
    grads_and_vars = []
    for gradient, variable in zip(adv_gradient_accumulator.gradients, adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    adv_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_encoder_classification_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)     
      encoder_classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_encoder_classification_loss, None)   
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, encoder_classification_loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  def _adv_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_adv_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())

  ### Running one step to compile graph
  _, _, _ = next(train_data_flow)

  ### Initialize weights or update if needed for Continual Learning
  if config.get("continual_learning", False):
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,
                        trackables={"model":model},
                        model_key="model")
                        
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _encoder_classification_loss = []
  _number_examples = []

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, encoder_classification_loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _encoder_classification_loss.append(encoder_classification_loss)
        _number_examples.append(num_examples)
      _step()  
      _adv_step()
      step = optimizer.iterations.numpy() // 2
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Encoder_classification_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_encoder_classification_loss), np.sum(_number_examples), elapsed)
        _loss = []  
        _encoder_classification_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def proxy_distance(config,
          optimizer,          
          learning_rate,
          model,  
          source_file,
          target_file,
          training_domain,
          eval_file,
          eval_domain,
          test_file,
          test_domain,
          strategy,  
          checkpoint_manager,
          checkpoint,
          save_dir,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    checkpoint_path = checkpoint_manager.latest_checkpoint  
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  output_dir = os.path.join(config["model_dir"],save_dir)
  new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
  
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  
  print("There are %d in-domain corpora"%len(source_file))
  print("batch type: ", batch_type)

  train_dataset = create_trainining_dataset(strategy, model, training_domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    e, logits = model.classification_on_top_encoder(source, training=True)
    tf.print("logits: ", logits)
    training_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], logits)    
    #variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name or "encoder" in var.name or "My_inputter_0" in var.name]
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0] 
    return tf.reduce_mean(training_loss), num_examples

  def _apply_gradients():
    #variables = model.trainable_variables
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    #variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name or "encoder" in var.name or "My_inputter_0" in var.name]
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []      

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % eval_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        new_checkpoint_manager.save(checkpoint_number=step)
        checkpoint_path = new_checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,i in zip(eval_file, eval_domain):
          domain_predict(src, model, checkpoint_path, checkpoint, i, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
      tf.summary.flush()
      if step > train_steps:
        break
  errors = []
  for src,i in zip(test_file, test_domain):
    accuracy = domain_predict(src, model, checkpoint_manager, checkpoint, i, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    errors.append(1-accuracy)
  return 2 * (1 - 2 * np.mean(errors))

def add_vocab(config,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      reported_loss = loss[0] / loss[2]
    else:
      _, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  train_data_flow = iter(_train_forward())
  _,_ = next(train_data_flow)    
  step = optimizer.iterations.numpy()
  tf.get_logger().info("Saving checkpoint for step %d", step)
  for v in model.trainable_variables:
    if "_embedding" in v.name:
      v.assign(tf.cast(tf.concat([v, tf.Variable(np.zeros(1,512),dtype=v.dtype)],0),v.dtype), validate_shape=False)

  checkpoint_manager.save(checkpoint_number=step)
  return checkpoint_manager.latest_checkpoint

def averaged_checkpoint_translate(config, source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              length_penalty,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=5,
              max_count=3):
  
  # Create the inference dataset.
  from os import path
  if not path.exists(path.join("%s/averaged_checkpoint"%config["model_dir"],"ckpt-200000.data-00000-of-00002")):
    new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=max_count,
                        model_key="model")
    checkpoint.restore(new_checkpoint_manager.latest_checkpoint)
    tf.get_logger().info("Evaluating model %s", new_checkpoint_manager.latest_checkpoint)
  else:
    checkpoint_path = path.join("%s/averaged_checkpoint"%config["model_dir"],"ckpt-200000")
    checkpoint.restore(checkpoint_path)
    tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv11","residualv7","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"]], source_length)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv15","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv6","residualv13","residualv12","residualv11","residualv7","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score