import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src")
parser.add_argument("--tgt")
parser.add_argument("--label")
parser.add_argument("--n_clusters")

args = parser.parse_args()

n_clusters = int(args.n_clusters)
src_path = args.src
tgt_path = args.tgt
label_path = args.label
src_files = [open(src_path+".cluster.%d"%i,"w") for i in range(n_clusters)]
tgt_files = [open(tgt_path+".cluster.%d"%i,"w") for i in range(n_clusters)]
src_tag_files = [open(src_path+".cluster.%d.tagged"%i,"w") for i in range(n_clusters)]
tgt_tag_files = [open(tgt_path+".cluster.%d.tagged"%i,"w") for i in range(n_clusters)]

with open(path,"r") as f1:
    with open(label_path,"r") as f2:
        ls = [l.strip() for l in f1.readlines()]
        domains = [int(l.strip()) for l in f2.readlines()]
        for l, domain in zip(ls,domains):
            print(l,file=src_files[domain])
            print(l,file=tgt_files[domain])
            print("Domain=%d %s"%(domain,l),file=src_tag_files[domain])
            print("Domain=%d %s"%(domain,l),file=tgt_tag_files[domain])

[f.close() for f in files]
[f.close() for f in tag_files]