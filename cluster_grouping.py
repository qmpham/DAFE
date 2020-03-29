import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src")
parser.add_argument("--label")
parser.add_argument("--n_clusters")

args = parser.parse_args()

n_clusters = int(args.n_clusters)
path = args.src
label_path = args.label
files = [open(path+".cluster.%d"%i,"w") for i in range(n_clusters)]
tag_files = [open(path+".cluster.%d.tag.0"%i,"w") for i in range(n_clusters)]

with open(path,"r") as f1:
    with open(label_path,"r") as f2:
        ls = [l.strip() for l in f1.readlines()]
        domains = [int(l.strip()) for l in f2.readlines()]
        for l, domain in zip(ls,domains):
            print(l,file=files[domain])
            print(domain,file=tag_files[domain])

print(files)
[f.closed() for f in files]
[f.closed() for f in tag_files]