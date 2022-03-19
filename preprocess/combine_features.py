import numpy as np
import os
import glob
import pandas as pd
import json

dataset = '50salads'
feat_name = '50salads_vit_features_splt1'
feat_root = './data/'+dataset+'/features_dir/'+feat_name
final_dir = 'combined_feat'
final_root = os.path.join(feat_root, final_dir)
feats = glob.glob(feat_root + '/*')
feats = [x for x in feats if x.endswith('.npy')]
if not os.path.exists(final_root):
    os.mkdir(final_root)

with open("./v_vlen_" + dataset + ".json", 'r') as f:
    v_vlen = json.load(f)
df = pd.DataFrame(feats, columns=['paths'])
df['vid'] = [d.rsplit('/', 1)[1].rsplit('_', 1)[0] for d in df.paths]
df['ind'] = [d.rsplit('_', 1)[1][:-4] for d in df.paths]
df['ind'] = df['ind'].astype(int)
for name, group in df.groupby('vid'):
    group.sort_values('ind', inplace=True)
    vlen = v_vlen[name]
    result = np.zeros((vlen, 768))
    for index, row in group.iterrows():
        tfeat = np.load(row.paths)
        if index == group.index[-1]:
            diff = vlen - row.ind
            result[row.ind:, :] = tfeat[:diff, :]
        else:
            result[row.ind:row.ind + 32, :] = tfeat
    np.save(os.path.join(final_root, name + '.npy'), result)
    print(name + ' is combined.')
