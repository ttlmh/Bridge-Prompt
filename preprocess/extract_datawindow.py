import numpy as np
import os

modes = ['train', 'test']
n_split = 1
num_frames = 16
overlap = [1, 1, 0.5]
dss = [1, 2, 4]
root = './data/gtea/'
frame_dir = './data/gtea/frames/'
for mode in modes:
    txt_path = 'splits/' + mode + '.split' + str(n_split) + '.bundle'
    train_split = []
    with open(os.path.join(root, txt_path), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('.')[0]
            train_split.append(line)
    new_train_list = []
    for i in range(len(dss)):
        for dat in train_split:
            vpath = os.path.join(frame_dir, dat)
            vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
            start_idxs = np.arange(0, vlen, int(num_frames * overlap[i] * dss[i]))
            for idx in start_idxs:
                new_train_list.append([dat, idx, dss[i]])
    np.save(f'./data/gtea/splits/'+mode+f'_split{n_split}_nf{num_frames}_ol{overlap}_ds{dss}.npy',
            np.array(new_train_list))
