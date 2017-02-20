import numpy as np
import numpy.random as npr
import os

data_dir='../data/wider'
anno_file = os.path.join(data_dir,"anno.txt")

size = 48

if size == 12:
    net = "pnet"
elif size == 24:
    net = "rnet"
elif size == 48:
    net = "onet"

with open(os.path.join(data_dir,'%s/pos_%s.txt'%(net, size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir,'%s/neg_%s.txt'%(net, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir,'%s/part_%s.txt'%(net, size)), 'r') as f:
    part = f.readlines()
dir_path=os.path.join(data_dir,'imglists')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(os.path.join(dir_path,"train_%s.txt"%( size)), "w") as f:
    nums=[len(neg),len(pos),len(part)]
    ratio=[3,1,1]
    base_num=min(nums)
    base_num=250000
    print len(neg),len(pos),len(part),base_num
    neg_keep = npr.choice(len(neg), size=base_num*3,replace=True)
    pos_keep = npr.choice(len(pos), size=base_num,replace=True)
    part_keep = npr.choice(len(part), size=base_num,replace=True)

    for i in pos_keep:
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
