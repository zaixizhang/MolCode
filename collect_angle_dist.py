import pickle
from config import conf
from runner import Runner
import torch
#from utils import check_validity
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
import random
from tqdm import tqdm

mols = Chem.SDMolSupplier('./qm9/gdb9.sdf', removeHs=False, sanitize=False)
split_path = 'qm9/split.npz'
bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
idxs = np.load(split_path)
subset_idxs = idxs['train_idx'].tolist()
#subset_idxs = subset_idxs[:10000]
bond_dist = {}
angle_dist = {}
for idx in tqdm(subset_idxs):
    mol = mols[idx]
    n_atoms = mol.GetNumAtoms()
    pos = mols.GetItemText(idx).split('\n')[4:4 + n_atoms]
    position = np.array([[float(x) for x in line.split()[:3]] for line in pos], dtype=np.float32)
    atom_type = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    bond_tmp=[]
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_tmp.append([start,end])
    for i in range(len(bond_tmp)-1):
        for j in range(i+1,len(bond_tmp)):
            if set(bond_tmp[i])&set(bond_tmp[j])!=set():
                pivotal = list(set(bond_tmp[i])&set(bond_tmp[j]))[0]
                p1 = list(set(bond_tmp[i])-set({pivotal}))[0]
                p2 = list(set(bond_tmp[j])-set({pivotal}))[0]
                z1, z2 = atom_type[p1], atom_type[p2]
                z1, z2 = min(z1, z2), max(z1, z2)
                if not (z1, z2, atom_type[pivotal]) in angle_dist:
                    angle_dist[(z1, z2, atom_type[pivotal])] = []
                vec1=position[p1] - position[pivotal]
                vec2 = position[p2] - position[pivotal]
                product = np.dot(np.array(vec1),np.array(vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                angle_dist[(z1, z2, atom_type[pivotal])].append(np.arccos(product)/np.pi*180)
        '''
        bond_type = bond_to_type[bond.GetBondType()]
        z1, z2 = atom_type[start], atom_type[end]
        z1, z2 = min(z1, z2), max(z1, z2)
        if not (z1, z2, bond_type) in bond_dist:
            bond_dist[(z1, z2, bond_type)] = []
        bond_dist[(z1, z2, bond_type)].append(np.linalg.norm(position[start] - position[end]))'''
np.save('/zaixizhang/exp_gen/angle_dist.npy', angle_dist)
