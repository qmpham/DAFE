import argparse
import numpy as np
import pickle
def proxy_distance(emb_files_domain_A, emb_files_domain_B, tst_emb_files_domain_A, tst_emb_files_domain_B, max_iter):
  from sklearn.svm import SVC
  emb_files_domain_A_list = []
  for emb_file in emb_files_domain_A:
    emb_storage = np.load(emb_file)
    embs = emb_storage["sentence_embeddings"]
    emb_files_domain_A_list.append(embs)
  X_A = np.concatenate(emb_files_domain_A_list, 0)
  Y_A = np.zeros((X_A.shape[0]))
  emb_files_domain_B_list = []
  for emb_file in emb_files_domain_B:
    emb_storage = np.load(emb_file)
    embs = emb_storage["sentence_embeddings"]
    emb_files_domain_B_list.append(embs)

  X_B = np.concatenate(emb_files_domain_B_list, 0)
  Y_B = np.ones((X_B.shape[0]))
  X = np.concatenate([X_A, X_B],0)
  Y = np.concatenate([Y_A, Y_B])
  print("Input shape: ", X.shape)
  clf = SVC(class_weight="balanced")
  clf.fit(X,Y)
  
  tst_emb_files_domain_A_list = []
  for emb_file in tst_emb_files_domain_A:
    emb_storage = np.load(emb_file)
    embs = emb_storage["sentence_embeddings"]
    tst_emb_files_domain_A_list.append(embs)
  tst_X_A = np.concatenate(tst_emb_files_domain_A_list, 0)
  tst_Y_A = np.zeros((tst_X_A.shape[0]))
  tst_emb_files_domain_B_list = []
  for emb_file in tst_emb_files_domain_B:
    emb_storage = np.load(emb_file)
    embs = emb_storage["sentence_embeddings"]
    tst_emb_files_domain_B_list.append(embs)

  tst_X_B = np.concatenate(tst_emb_files_domain_B_list, 0)
  tst_Y_B = np.ones((tst_X_B.shape[0]))
  tst_X = np.concatenate([tst_X_A, tst_X_B],0)
  tst_Y = np.concatenate([tst_Y_A, tst_Y_B])

  score = clf.score(tst_X, tst_Y)
  print(2 * (2*score - 1))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--emb_files_domain_A", nargs="+")
parser.add_argument("--emb_files_domain_B", nargs="+")
parser.add_argument("--max_iter")
parser.add_argument("--tst_emb_files_domain_A", nargs="+")
parser.add_argument("--tst_emb_files_domain_B", nargs="+")
args = parser.parse_args()

emb_files_domain_A = args.emb_files_domain_A
emb_files_domain_B = args.emb_files_domain_B
tst_emb_files_domain_A = args.tst_emb_files_domain_A
tst_emb_files_domain_B = args.tst_emb_files_domain_B
max_iter= int(args.max_iter)
proxy_distance(emb_files_domain_A, emb_files_domain_B, tst_emb_files_domain_A, tst_emb_files_domain_B, max_iter)