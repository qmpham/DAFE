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
from encoders.self_attention_encoder import Multi_domain_SelfAttentionEncoder, Multi_domain_SelfAttentionEncoder_v2, Multi_domain_SelfAttentionEncoder_v0
from decoders.self_attention_decoder import Multi_domain_SelfAttentionDecoder, Multi_domain_SelfAttentionDecoder_v0, Multi_domain_SelfAttentionDecoder_v2, Multi_domain_SelfAttentionDecoder_v1, Multi_domain_SelfAttentionDecoder_v5
from layers.layers import Multi_domain_FeedForwardNetwork_v3
import numpy as np
from utils.dataprocess import merge_map_fn, create_meta_trainining_dataset, create_trainining_dataset, create_multi_domain_meta_trainining_dataset
from opennmt.utils import BLEUScorer
from opennmt.inputters.text_inputter import WordEmbedder
from utils.utils_ import variance_scaling_initialier, MultiBLEUScorer
import task
from layers.layers import Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, DAFE, Multi_domain_FeedForwardNetwork_v1
def main():
  devices = tf.config.experimental.list_logical_devices(device_type="GPU")
  print(devices)
  strategy = tf.distribute.MirroredStrategy(devices=[d.name for d in devices])
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train", "trainv3", "metatrainv12", "trainv2", "trainv8", "translate", "translatev2", "translatev3", "metatrainv9", "metatrainv11", "debug","metatrainv1", "metatrainv2", "metatrainv3", "inspect", "metatrainv5", "metatrainv6", "metatrainv7", "metatrainv8", "metatrainv10", "finetune"], help="Run type.")
  parser.add_argument("--config", required=True , help="configuration file")
  parser.add_argument("--src")
  parser.add_argument("--output", default="trans")
  parser.add_argument("--domain", default=0)
  parser.add_argument("--ref", default=None)
  parser.add_argument("--maxcount",default=3)
  args = parser.parse_args()
  print("Running mode: ", args.run)
  config_file = args.config
  with open(config_file, "r") as stream:
      config = yaml.load(stream)
  if not os.path.exists(os.path.join(config["model_dir"],"eval")):
    os.makedirs(os.path.join(config["model_dir"],"eval"))
  data_config = {
      "source_vocabulary": config["src_vocab"],
      "target_vocabulary": config["tgt_vocab"]
  }
  experiment = config.get("experiment","residual")
  print("running experiment: ", experiment)
  ADAP_layer_stopping_gradient = config.get("ADAP_layer_stopping_gradient",False)
  print("ADAP_layer_stopping_gradient: ", ADAP_layer_stopping_gradient)
  num_domain_units = config.get("num_domain_units",128)
  num_domains = config.get("num_domains", 6)
  if experiment=="residual":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder(
        num_layers=6,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder(
        num_layers=6,
        num_domains=6,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv2":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv5":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv0":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v0(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v1),
    decoder=Multi_domain_SelfAttentionDecoder_v0(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v1))
  elif experiment=="residualv1":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv3":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder_v5(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="ldr":
    model = LDR_SequenceToSequence(
    source_inputter=LDR_inputter(embedding_size=config.get("ldr_embedding_size",464), num_domains=config.get("num_domains", 8), num_domain_units=config.get("num_domain_units", 8)),
    target_inputter=LDR_inputter(embedding_size=config.get("ldr_embedding_size",464), num_domains=config.get("num_domains", 8), num_domain_units=config.get("num_domain_units", 8)),
    encoder=onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="baseline":
    model = LDR_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="baselinev1":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[0.0] * 6),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[0.0] * 6))
  elif experiment=="pretrain":
    return
  
  learning_rate = onmt.schedules.ScheduleWrapper(schedule=onmt.schedules.NoamDecay(scale=1.0, model_dim=512, warmup_steps=4000), step_duration= config.get("step_duration",16))
  meta_train_optimizer = tf.keras.optimizers.SGD(0.0001)
  meta_test_optimizer = tfa.optimizers.LazyAdam(learning_rate)
  checkpoint = tf.train.Checkpoint(model=model, optimizer=meta_test_optimizer)   
  model.initialize(data_config)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, config["model_dir"], max_to_keep=5)
  ######
  model.params.update({"label_smoothing": 0.1})
  model.params.update({"average_loss_in_time": True})
  model.params.update({"beam_width": 5})
  ######
  if args.run == "inspect":
    task.model_inspect(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  if args.run == "metatrainv7":
    task.meta_train_v7(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv8":
    task.meta_train_v8(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
  elif args.run == "trainv8":
    task.train_v8(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
  elif args.run == "metatrainv10":
    task.meta_train_v10(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv11":
    task.meta_train_v11(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv12":
    task.meta_train_v12(config, meta_train_optimizer, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv9":
    task.meta_train_v9(config, meta_test_optimizer, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv6":
    task.meta_train_v6(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv5":
    task.meta_train_v5(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv2":
    task.meta_train_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv3":
    task.meta_train_v3(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv1":
    task.meta_train_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run =="train":
    task.train(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run =="trainv2":
    task.train_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "trainv3":
    task.train_v3(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "translate":
    model.build(None)
    print("translate in domain %d"%(int(args.domain)))
    task.translate(args.src, args.ref, model, checkpoint_manager,
              checkpoint, int(args.domain), args.output, length_penalty=0.6, experiment=experiment)
  elif args.run=="translatev2":
    model.create_variables()
    print("translate in domain %d"%(int(args.domain)))
    task.averaged_checkpoint_translate(config, args.src, args.ref, model, checkpoint_manager,
              checkpoint, int(args.domain), args.output, length_penalty=0.6, experiment=experiment, max_count=args.maxcount)
  elif args.run=="translatev3":
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):
      output_file = "%s.trans"%src_file.strip().split("/")[-1]
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      task.averaged_checkpoint_translate(config, src_file, None, model, checkpoint_manager,
              checkpoint, int(domain), output_file, length_penalty=0.6, experiment=experiment, max_count=translate_config.get("max_count",3))
  elif args.run == "finetune":
    task.finetuning(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "debug":
    task.debug(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
if __name__ == "__main__":
  main()
