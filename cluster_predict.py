import argparse
import numpy as np
import pickle
def kmeans_clustering(emb_files, n_clusters, kmeans_save_path, labels_ouput_path):
  from sklearn.cluster import KMeans
  emb_list = []
  for emb_file in emb_files:
    emb_storage = np.load(emb_file)
    embs = emb_storage["sentence_embeddings"]
    emb_list.append(embs)
  
  X = np.concatenate(emb_list,0)
  print("Input shape: ", X.shape)
  print("n_cluster: ", n_clusters)
  kmeans = pickle.load(open(kmeans_save_path, 'rb'))

  label_predictions = kmeans.predict(X)
  
  with open(labels_ouput_path, "w") as f:
    for l in label_predictions:
      print(l,file=f)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--emb_files", nargs="+")
parser.add_argument("--n_clusters", default=30)
parser.add_argument("--kmeans_save_path")
parser.add_argument("--output", default="trans")

args = parser.parse_args()

kmeans_save_path = args.kmeans_save_path
emb_files = args.emb_files
n_clusters = int(args.n_clusters)
labels_ouput_path = args.output
kmeans_clustering(emb_files, n_clusters, kmeans_save_path, labels_ouput_path)