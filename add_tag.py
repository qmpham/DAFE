import sys

with open(sys.argv[1],"r") as f_r:
    with open(sys.argv[2],"r") as f_tag:
        with open(sys.argv[1]+".30.clusters.tagged","w") as f_w:
            ls = [l.strip() for l in f_r.readlines()]
            tags = [int(t.strip()) for t in f_tag.readlines()]
            for l, tag in zip(ls,tags):
                print("Domain=%d %s"%(tag,l),file=f_w)