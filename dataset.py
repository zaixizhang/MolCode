import numpy as np
import torch
from torch.utils.data import Dataset
import os
import networkx as nx
from networkx.algorithms import tree
from math import pi
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
import pickle
from rdkit import Chem


def collate_mols(mol_dicts):
    data_batch = {}

    for key in ['atom_type', 'position', 'new_atom_type', 'new_dist', 'new_angle', 'new_torsion', 'cannot_focus', 'focus_bond', 'new_atom_bond', 'new_bond_type']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    data_batch['new_atom_bond1'] = torch.cat([mol_dict['new_atom_bond'][1:] for mol_dict in mol_dicts], dim=0)
    data_batch['new_atom_bond2'] = torch.cat([mol_dict['new_atom_bond'][3:] for mol_dict in mol_dicts], dim=0)
    '''
    for key in ['adj_array', 'atom_array']:
        data_batch[key] = torch.cat([mol_dict[key].unsqueeze(0) for mol_dict in mol_dicts])
    '''
    num_steps_list = torch.tensor([0]+[len(mol_dicts[i]['new_atom_type']) for i in range(len(mol_dicts)-1)])
    batch_idx_offsets = torch.cumsum(num_steps_list, dim=0)
    repeats = torch.tensor([len(mol_dict['batch']) for mol_dict in mol_dicts])
    batch_idx_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
    batch_offseted = torch.cat([mol_dict['batch'] for mol_dict in mol_dicts], dim=0) + batch_idx_repeated_offsets
    data_batch['batch'] = batch_offseted
    repeats = torch.tensor([len(mol_dict['edge_index']) for mol_dict in mol_dicts])
    edge_index_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
    edge_offseted = torch.cat([mol_dict['edge_index'] for mol_dict in mol_dicts], dim=0) + edge_index_repeated_offsets
    data_batch['edge_index'] = edge_offseted

    num_atoms_list = torch.tensor([0]+[len(mol_dicts[i]['atom_type']) for i in range(len(mol_dicts)-1)])
    atom_idx_offsets = torch.cumsum(num_atoms_list, dim=0)
    for key in ['focus', 'c1_focus', 'c2_c1_focus']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
        atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets[:,None]
        data_batch[key] = atom_offseted

    repeats = torch.tensor([len(mol_dict['focus_index']) for mol_dict in mol_dicts])
    atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
    atom_offseted = torch.cat([mol_dict['focus_index'] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets
    data_batch['focus_index'] = atom_offseted

    return data_batch


class QM9Gen(Dataset):
    def __init__(self, cutoff, root_path, subset_idxs=None, atomic_num_to_type={1:0, 6:1, 7:2, 8:3, 9:4}):
        super().__init__()
        #self.mols = Chem.SDMolSupplier(os.path.join(root_path, 'gdb9.sdf'), removeHs=False, sanitize=False)
        self.mols= pickle.load(open(root_path, 'rb'))
        print('data load!')
        self.atomic_num_to_type = atomic_num_to_type
        self.bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
        self.cutoff = cutoff
        self.num_max_node = 29
        self.perm = 'minimum_spanning_tree'
        if subset_idxs is not None:
            self.subset_idxs = subset_idxs
    
    def _get_subset_idx(self, split_file, mode):
        idxs = np.load(split_file)
        if mode == 'train':
            self.subset_idxs = idxs.tolist()
        elif mode == 'val':
            self.subset_idxs = idxs.tolist()
        elif mode == 'test':
            self.subset_idxs = idxs.tolist()
    
    def _get_mols(self, index):
        if hasattr(self, 'subset_idxs'):
            idx = int(self.subset_idxs[index])
        else:
            idx = index

        mol = self.mols[idx]
        n_atoms = mol.GetNumAtoms()
        pos = self.mols.GetItemText(idx).split('\n')[4:4+n_atoms]
        position = np.array([[float(x) for x in line.split()[:3]] for line in pos], dtype=np.float32)
        atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
        '''
        mol = self.mols[idx].rdmol
        Chem.Kekulize(mol)
        n_atoms = mol.GetNumAtoms()
        position = np.array(self.mols[idx].pos, dtype=np.float32)
        atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()])'''

        #2D atom type
        atom_array = np.zeros((self.num_max_node, len(self.atomic_num_to_type)), dtype=np.float32)
        atom_idx = 0
        '''
        for atom in mol.GetAtoms():
            atom_array[atom_idx, self.atomic_num_to_type[atom.GetAtomicNum()]] = 1
            atom_idx += 1
        '''
        con_mat = np.zeros([n_atoms, n_atoms], dtype=int)
        adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_to_type[bond.GetBondType()]
            con_mat[start, end] = bond_type
            con_mat[end, start] = bond_type
            adj_array[bond_type - 1, start, end] = 1.0
            adj_array[bond_type - 1, end, start] = 1.0
        adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
        adj_array += np.eye(self.num_max_node)
        
        if atom_type[0] != 1:
            first_carbon = np.nonzero(atom_type == 1)[0][0]
            perm = np.arange(len(atom_type))
            perm[0] = first_carbon
            perm[first_carbon] = 0
            atom_type, position = atom_type[perm], position[perm]
            con_mat = con_mat[perm][:, perm]
            perm_extend = np.concatenate([perm, np.arange(len(atom_type), self.num_max_node)])
            atom_array = atom_array[perm_extend]
            for i in range(4):
                adj_array[i] = adj_array[i][perm_extend][:, perm_extend]

        return torch.tensor(atom_type), torch.tensor(position), torch.tensor(con_mat), torch.tensor(np.sum(con_mat, axis=1)), torch.tensor(atom_array), torch.tensor(adj_array)

    def _bfs_seq(self, G, start_id):
        dictionary = dict(nx.bfs_successors(G, start_id))
        start = [start_id]
        output = [start_id]
        focus_node_id = []
        while len(start) > 0:
            next_vertex = []
            while len(start) > 0:
                current = start.pop(0)
                neighbor = dictionary.get(current)
                if neighbor is not None:
                    next_vertex = next_vertex + neighbor
                    focus_node_id = focus_node_id + [current for _ in range(len(neighbor))]
            output = output + next_vertex
            start = next_vertex
        return output, focus_node_id

    def get_masks(self, num_node):
        node_masks1 = torch.zeros([num_node-1, self.num_max_node]).bool()
        adj_masks1 = torch.zeros([num_node-1, self.num_max_node, self.num_max_node]).bool()
        for i in range(num_node-1):
            node_masks1[i][:i+1] = 1
            adj_masks1[i][:i+1, :i+1] = 1
        return node_masks1, adj_masks1

    
    def __len__(self):
        if hasattr(self, 'subset_idxs'):
            return len(self.subset_idxs)
        else:
            return len(self.mols)

    def __getitem__(self, index):
        atom_type, position, con_mat, atom_valency, atom_array, adj_array = self._get_mols(index)
        num_atom = len(atom_type)
        atom_array = torch.tensor(atom_array)
        adj_array = torch.tensor(adj_array)
        #build tree based on 3d distances
        squared_dist = torch.sum(torch.square(position[:,None,:] - position[None,:,:]), dim=-1)
        if self.perm == 'minimum_spanning_tree':
            nx_graph = nx.from_numpy_matrix(squared_dist.numpy())
            edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False))
            focus_node_id, target_node_id = zip(*edges)
            node_perm = torch.cat((torch.tensor([0]), torch.tensor(target_node_id)))
            perm_extend = torch.cat([node_perm, torch.arange(num_atom, self.num_max_node)])
        elif self.perm == 'bfs':
            #build tree based on bfs
            pure_adj = np.sum(adj_array[:3].numpy(), axis=0)[:num_atom, :num_atom]
            G = nx.from_numpy_matrix(np.asmatrix(pure_adj))
            start_idx = 0
            node_perm, focus_node_id = self._bfs_seq(G, start_idx)
            node_perm = np.array(node_perm)
            perm_extend = np.concatenate([node_perm, np.arange(num_atom, self.num_max_node)])

        atom_array = atom_array[perm_extend]
        for i in range(4):
            adj_array[i] = adj_array[i][perm_extend][:, perm_extend]
        #node_masks1, adj_masks1 = self.get_masks(num_atom)
        #adj = adj_array[:3]
        #x = torch.where(node_masks1.view(num_atom-1, self.num_max_node, 1), atom_array.unsqueeze(0).repeat(num_atom-1, 1, 1), torch.zeros([1], device=atom_array.device))
        #adj = torch.where(adj_masks1.view(num_atom-1, 1, self.num_max_node, self.num_max_node), adj.unsqueeze(0).repeat(num_atom-1, 1, 1, 1), torch.zeros([1], device=atom_array.device))

        position = position[node_perm]
        atom_type = atom_type[node_perm]
        con_mat = con_mat[node_perm][:,node_perm]
        squared_dist = squared_dist[node_perm][:,node_perm]
        atom_valency = atom_valency[node_perm]
        # print(con_mat)

        focus_node_id = torch.tensor(focus_node_id)
        node_perm = torch.tensor(node_perm)
        steps_focus = torch.nonzero(focus_node_id[:,None] == node_perm[None,:])[:,1]
        steps_c1_focus, steps_c2_c1_focus = torch.empty([0,2], dtype=int), torch.empty([0,3], dtype=int)
        steps_focus_bond, steps_new_atom_bond = torch.empty([0,1], dtype=int), torch.empty([0,1], dtype=int)
        steps_batch, steps_position, steps_atom_type = torch.empty([0,1], dtype=int), torch.empty([0,3], dtype=position.dtype), torch.empty([0,1], dtype=atom_type.dtype)
        steps_cannot_focus = torch.empty([0,1], dtype=float)
        steps_dist, steps_angle, steps_torsion = torch.empty([0,1], dtype=float), torch.empty([0,1], dtype=float), torch.empty([0,1], dtype=float)
        idx_offsets = torch.cumsum(torch.arange(len(atom_type) - 1), dim=0)
        edge_index = torch.tensor([0, 1, 1]).view(-1,1)
        new_bond_type = torch.empty([0,1], dtype=int)
        
        for i in range(min(len(atom_type), self.num_max_node) - 1):
            partial_con_mat = con_mat[:i+1, :i+1]
            valency_sum = partial_con_mat.sum(dim=1, keepdim=True)
            steps_cannot_focus = torch.cat((steps_cannot_focus, (valency_sum == atom_valency[:i+1, None]).float()))

            one_step_focus = steps_focus[i]
            one_step_focus_bond = con_mat[one_step_focus, :i+1]
            one_step_new_atom_bond = con_mat[i+1,:i+1]

            focus_pos, new_pos = position[one_step_focus], position[i+1]
            one_step_dis = torch.norm(new_pos - focus_pos)
            steps_dist = torch.cat((steps_dist, one_step_dis.view(1,1)))
            
            if i > 0:
                mask = torch.ones([i+1], dtype=torch.bool)
                mask[one_step_focus] = False
                c1_dists = squared_dist[one_step_focus, :i+1][mask]
                one_step_c1 = torch.argmin(c1_dists)
                if one_step_c1 >= one_step_focus:
                    one_step_c1 += 1
                steps_c1_focus = torch.cat((steps_c1_focus, torch.tensor([one_step_c1, one_step_focus]).view(1,2) + idx_offsets[i]))

                c1_pos = position[one_step_c1]
                a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                one_step_angle = torch.atan2(b,a)
                steps_angle = torch.cat((steps_angle, one_step_angle.view(1,1)))

                if i > 1:
                    mask[one_step_c1] = False
                    c2_dists = squared_dist[one_step_c1, :i+1][mask]
                    one_step_c2 = torch.argmin(c2_dists)
                    if one_step_c2 >= min(one_step_c1, one_step_focus):
                        one_step_c2 += 1
                        if one_step_c2 >= max(one_step_c1, one_step_focus):
                            one_step_c2 += 1
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, torch.tensor([one_step_c2, one_step_c1, one_step_focus]).view(1,3) + idx_offsets[i]))

                    c2_pos = position[one_step_c2]
                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion.view(1,1)))
                    edge_index = torch.cat((edge_index, torch.tensor([i]).repeat(3).view(-1,1)))
                    
            one_step_position = position[:i+1]
            steps_position = torch.cat((steps_position, one_step_position), dim=0)
            one_step_atom_type = atom_type[:i+1]
            steps_atom_type = torch.cat((steps_atom_type, one_step_atom_type.view(-1,1)))
            steps_batch = torch.cat((steps_batch, torch.tensor([i]).repeat(i+1).view(-1,1)))
            steps_focus_bond = torch.cat((steps_focus_bond, one_step_focus_bond.view(-1,1)))
            steps_new_atom_bond = torch.cat((steps_new_atom_bond, one_step_new_atom_bond.view(-1, 1)))
            if i ==0:
                new_bond_type = torch.cat((new_bond_type, con_mat[one_step_focus,1].view(-1, 1)))
            elif i ==1:
                new_bond_type = torch.cat((new_bond_type, con_mat[one_step_c1, 2].view(-1, 1), con_mat[one_step_focus, 2].view(-1, 1)))
            else:
                new_bond_type = torch.cat((new_bond_type, con_mat[one_step_c2, i+1].view(-1, 1), con_mat[one_step_c1, i+1].view(-1, 1), con_mat[one_step_focus, i+1].view(-1, 1)))
        
        steps_focus += idx_offsets
        steps_new_atom_type = atom_type[1:]
        steps_torsion[steps_torsion <= 0] += 2 * pi
        focus_index = torch.cat((steps_focus[0].view(-1,1), steps_c1_focus[0].view(-1,1), steps_c2_c1_focus.view(-1,1)))

        data_batch = {}
        data_batch['atom_type'] = steps_atom_type.view(-1)
        data_batch['position'] = steps_position
        data_batch['batch'] = steps_batch.view(-1)
        data_batch['focus'] = steps_focus[:,None]
        data_batch['c1_focus'] = steps_c1_focus
        data_batch['c2_c1_focus'] = steps_c2_c1_focus
        data_batch['new_atom_type'] = steps_new_atom_type.view(-1)
        data_batch['new_dist'] = steps_dist
        data_batch['new_angle'] = steps_angle
        data_batch['new_torsion'] = steps_torsion
        data_batch['cannot_focus'] = steps_cannot_focus.view(-1).float()
        #data_batch['adj_array'] = adj_array
        #data_batch['atom_array'] = atom_array
        #data_batch['x'] = x
        #data_batch['adj'] = adj
        data_batch['focus_bond'] = steps_focus_bond.view(-1)
        data_batch['new_atom_bond'] = steps_new_atom_bond.view(-1)
        data_batch['edge_index'] = edge_index.view(-1)
        data_batch['focus_index'] = focus_index.view(-1)
        data_batch['new_bond_type'] = new_bond_type.view(-1)

        return data_batch

