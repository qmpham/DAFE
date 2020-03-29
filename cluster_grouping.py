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

with open(src_path,"r") as f1:
    with open(tgt_path, "r") as f3:
        with open(label_path,"r") as f2:
            src_ls = [l.strip() for l in f1.readlines()]
            tgt_ls = [l.strip() for l in f3.readlines()]
            domains = [int(l.strip()) for l in f2.readlines()]
            for src_l, tgt_l, domain in zip(src_ls, tgt_ls, domains):
                print(l,file=src_files[domain])
                print(l,file=tgt_files[domain])
                print("Domain=%d %s"%(domain,src_l),file=src_tag_files[domain])
                print("Domain=%d %s"%(domain,tgt_l),file=tgt_tag_files[domain])

[f.close() for f in src_files]
[f.close() for f in tgt_files]
[f.close() for f in src_tag_files]
[f.close() for f in tgt_tag_files]