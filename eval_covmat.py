import os
import argparse
import pickle
import torch
import numpy as np
import multiprocessing as mp
from functools import partial
from torch_geometric.data import Data
from tqdm.auto import tqdm

from utils.covmat import get_rmsd_confusion_matrix

out_path = '/apdcephfs/private_zaixizhang/exp_gen/13/'
data_list = pickle.load(open('/apdcephfs/private_zaixizhang/exp_gen/13/0.30_0.50_0.50_0.50_0.30.pkl', 'rb'))

# Evaluator
cov_list = []
mat_list = []
threshold = 1.25
func = partial(get_rmsd_confusion_matrix, useFF=True)
pool = mp.Pool(4)
for confusion_mat in tqdm(pool.imap(func, data_list), total=len(data_list)):
    rmsd_ref_min = confusion_mat.min(-1)  # np (num_ref, )
    rmsd_cov_thres = (rmsd_ref_min<=threshold)
    mat_list.append(rmsd_ref_min.mean())
    cov_list.append(rmsd_cov_thres.mean())
    file_obj = open(os.path.join('/apdcephfs/private_zaixizhang/exp_gen/', 'generation2.txt'), 'a')
    file_obj.write('{} || {} \n'.format(rmsd_ref_min.mean(), rmsd_cov_thres.mean()))
    file_obj.close()
cov_mean, mat_mean = np.mean(cov_list),np.mean(mat_list)
print(cov_mean)
print(mat_mean)



