import argparse
import task

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("run", choices=["train", "kmeans", "sentence_encode", "train_wdc", "train_denny_britz", "train_ldr", "visualize", "experimental_translate", "trainv3", "dcote", "metatrainv12", "trainv13", "trainv2", "trainv12", "metatrainv15", "translatev1", "trainv8", "translate", "translatev2", "translatev3", "metatrainv9", "metatrainv11", "debug","metatrainv1", "metatrainv2", "metatrainv3", "inspect", "metatrainv5", "metatrainv6", "metatrainv7", "metatrainv8", "metatrainv10", "finetune"], help="Run type.")
parser.add_argument("--config", required=True , help="configuration file")
parser.add_argument("--src")
parser.add_argument("--emb_files", nargs="+")
parser.add_argument("--n_clusters", default=30)
parser.add_argument("--kmeans_save_path")

args = parser.parse_args()
print("Running mode: ", args.run)

kmeans_save_path = args.kmeans_save_path
emb_files = args.emb_files
n_clusters = args.n_clusters
labels_ouput_path = args.output
task.kmeans_clustering(emb_files, n_clusters, kmeans_save_path, labels_ouput_path)