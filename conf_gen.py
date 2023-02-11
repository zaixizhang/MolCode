import pickle
from config import conf
from runner import Runner
import torch
import numpy as np
from rdkit import Chem
import argparse
import os
import pickle
import torch
import networkx as nx
from rdkit.Chem.rdchem import BondType
from collections import defaultdict
import copy
from networkx.algorithms import tree


def generate(node_temp, dist_temp, angle_temp, torsion_temp, bond_temp):
    bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
    atomic_num_to_type = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    data_list = pickle.load(open('/apdcephfs/private_zaixizhang/data/qm9_processed/test_data.pkl', 'rb'))
    root_path = '/apdcephfs/private_zaixizhang/data/qm9_processed/'
    out_path = '/apdcephfs/private_zaixizhang/exp_gen/11/'
    runner = Runner(conf, root_path=root_path, out_path=out_path)
    runner.model.load_state_dict(torch.load('/apdcephfs/private_zaixizhang/exp_gen/11/best_valid1.pth', map_location='cuda:0'))
    out_list = []
    packed_data = defaultdict(list)
    new_data = []
    for i in range(len(data_list)):
        packed_data[data_list[i].smiles].append(data_list[i])
    for k, v in packed_data.items():
        data = copy.deepcopy(v[0])
        all_pos = []
        for i in range(len(v)):
            all_pos.append(v[i].pos)
        data.pos_ref = torch.cat(all_pos, 0)  # (num_conf*num_node, 3)
        data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
        # del data.pos
        if hasattr(data, 'totalenergy'):
            del data.totalenergy
        if hasattr(data, 'boltzmannweight'):
            del data.boltzmannweight
        new_data.append(data)

    for i in range(len(new_data)):
        data = new_data[i]
        n_atoms = data.atom_type.shape[0]
        node_type = torch.tensor([atomic_num_to_type[atom.item()] for atom in data.atom_type])
        con_mat = torch.zeros([n_atoms, n_atoms], dtype=int)
        Chem.Kekulize(data.rdmol)
        mol = data.rdmol
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond_to_type[bond.GetBondType()]
            con_mat[start, end] = bond_type
            con_mat[end, start] = bond_type

        if node_type[0] != 1:
            first_carbon = np.nonzero(node_type == 1)[0][0]
            perm = np.arange(len(node_type))
            perm[0] = first_carbon
            perm[first_carbon] = 0
            node_type = node_type[perm]
            con_mat = con_mat[perm][:, perm]
        position = data.pos
        squared_dist = torch.sum(torch.square(position[:, None, :] - position[None, :, :]), dim=-1)
        nx_graph = nx.from_numpy_matrix(squared_dist.numpy())
        edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False))
        focus_node_id, target_node_id = zip(*edges)
        node_perm = torch.cat((torch.tensor([0]), torch.tensor(target_node_id)))

        if len(node_perm) != n_atoms:
            continue
        node_perm = np.array(node_perm)

        node_type = node_type[node_perm]
        con_mat = con_mat[node_perm][:,node_perm]

        focus_node_id = torch.tensor(focus_node_id)
        node_perm = torch.tensor(node_perm)
        focuses = torch.nonzero(focus_node_id[:, None] == node_perm[None, :])[:, 1]

        num_gen = data.num_pos_ref.item()*2
        pos_gen = runner.generate_conf(num_gen, temperature=[node_temp, dist_temp, angle_temp, torsion_temp, bond_temp], node_type=node_type, con_mat=con_mat, focuses=focuses)
        data.pos_gen = pos_gen
        out_list.append(data)

    with open(out_path+'{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.pkl'.format(node_temp, dist_temp, angle_temp, torsion_temp, bond_temp),'wb') as f:
        pickle.dump(out_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of CoGen')
    parser.add_argument('--node', type=float, default=0.5,
                        help='node_temp')
    parser.add_argument('--edge', type=float, default=0.5,
                        help='edge_temp')
    parser.add_argument('--dist', type=float, default=0.3,
                        help='dist_temp')
    parser.add_argument('--angle', type=float, default=0.5,
                        help='angle_temp')
    parser.add_argument('--torsion', type=float, default=1.0,
                        help='torsion_temp')
    args = parser.parse_args()
    generate(args.node, args.dist, args.angle, args.torsion, args.edge)
    
