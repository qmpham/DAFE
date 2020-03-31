import argparse
import numpy as np
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--label",nargs="+")
parser.add_argument("--n_clusters")

args = parser.parse_args()

n_clusters = int(args.n_clusters)
label_paths = args.label
domain_numb = len(label_paths)
confusion_matrix = np.zeros([domain_numb, n_clusters])

for i in range(domain_numb):
    label_path = label_paths[i]
    with open(label_path,"r") as f2:
        domains = [int(l.strip()) for l in f2.readlines()]
        for domain in domains:
            confusion_matrix[i, domain] += 1
A = confusion_matrix/np.sum(confusion_matrix,0).reshape(1,-1)

for i in range(domain_numb):
    print("\t".join([str(p) for p in A[i,:]]))
