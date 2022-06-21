import pickle
from config import conf
from runner import Runner
import torch
from utils import check_validity, compute_bonds_mmd
import numpy as np
from rdkit import Chem
import argparse
import os

def generate(node_temp, dist_temp, angle_temp, torsion_temp, bond_temp):
    runner = Runner(conf)
    min_atoms = 4
    max_atoms = 35
    focus_th = 0.5
    num_gen = 1000
    out_path = '/apdcephfs/private_zaixizhang/exp_gen/7/'

    runner.model.load_state_dict(torch.load('/apdcephfs/private_zaixizhang/exp_gen/7/best_valid1.pth', map_location='cuda:0'))
    mol_dicts = runner.generate(num_gen, temperature=[node_temp, dist_temp, angle_temp, torsion_temp, bond_temp], max_atoms=max_atoms, min_atoms=min_atoms, focus_th=focus_th, add_final=False)
    results, valid_list, con_mat_list = check_validity(mol_dicts)
    print(results)
    if out_path is not None:
        file_obj = open(os.path.join(out_path, 'grid_search.txt'), 'a')
        file_obj.write('node_tmp {:.2f} dist_tmp {:.2f} angle_tmp {:.2f} torsion_tmp {:.2f} bond_tmp {:.2f}| Validity {:.4f}\n'.format(node_temp, dist_temp, angle_temp, torsion_temp, bond_temp, results['valid_ratio']))
        file_obj.close()
    with open(out_path+'{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.mol_dict'.format(node_temp, dist_temp, angle_temp, torsion_temp, bond_temp),'wb') as f:
        pickle.dump(mol_dicts, f)
    '''
    target_bond_dist = np.load('/apdcephfs/private_zaixizhang/exp_gen/target_dist.npy',allow_pickle=True)
    target_bond_dist = target_bond_dist.item()
    compute_bonds_mmd(mol_dicts, valid_list, con_mat_list, target_bond_dist, file_name='/apdcephfs/private_zaixizhang/exp_gen/generation1.txt')
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of CoGen')
    parser.add_argument('--node', type=float, default=0.5,
                        help='node_temp')
    parser.add_argument('--edge', type=float, default=0.5,
                        help='edge_temp')
    parser.add_argument('--dist', type=float, default=0.3,
                        help='dist_temp')
    parser.add_argument('--angle', type=float, default=0.4,
                        help='angle_temp')
    parser.add_argument('--torsion', type=float, default=1.0,
                        help='torsion_temp')
    args = parser.parse_args()
    generate(args.node, args.dist, args.angle, args.torsion, args.edge)

#with open('rand_gen/{}_mols.mol_dict'.format(epoch),'wb') as f:
    #pickle.dump(mol_dicts, f)
