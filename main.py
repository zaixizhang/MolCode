from config import conf
from runner import Runner
import os
import torch

out_path = '/apdcephfs/private_zaixizhang/exp_gen/1/'
root_path = '/apdcephfs/private_zaixizhang/data/drugs_processed/'
if not os.path.isdir(out_path):
    os.mkdir(out_path)

runner = Runner(conf, root_path = root_path, out_path=out_path)
print('Start Training!')
runner.train(root_path='./qm9', split_path='qm9/split.npz')

