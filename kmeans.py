import argparse
import task

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src")
parser.add_argument("--emb_files", nargs="+")
parser.add_argument("--n_clusters", default=30)
parser.add_argument("--kmeans_save_path")

args = parser.parse_args()

kmeans_save_path = args.kmeans_save_path
emb_files = args.emb_files
n_clusters = args.n_clusters
labels_ouput_path = args.output
task.kmeans_clustering(emb_files, n_clusters, kmeans_save_path, labels_ouput_path)