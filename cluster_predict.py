import argparse
import numpy as np
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
  with np.load(kmeans_save_path, allow_pickle=True) as data:
    print(data)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')
    kmeans.set_params(**data)

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