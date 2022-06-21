import pickle
from config import conf
from runner import Runner
import torch
#from utils import check_validity
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

mols = Chem.SDMolSupplier('./qm9/gdb9.sdf', removeHs=False, sanitize=False)
split_path = 'qm9/split.npz'
bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
idxs = np.load(split_path)
subset_idxs = idxs['train_idx'].tolist()
subset_idxs = subset_idxs[:10000]
bond_dist = {}
for idx in subset_idxs:
    mol = mols[idx]
    n_atoms = mol.GetNumAtoms()
    pos = mols.GetItemText(idx).split('\n')[4:4 + n_atoms]
    position = np.array([[float(x) for x in line.split()[:3]] for line in pos], dtype=np.float32)
    atom_type = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_to_type[bond.GetBondType()]
        z1, z2 = atom_type[start], atom_type[end]
        z1, z2 = min(z1, z2), max(z1, z2)
        if not (z1, z2, bond_type) in bond_dist:
            bond_dist[(z1, z2, bond_type)] = []
        bond_dist[(z1, z2, bond_type)].append(np.linalg.norm(position[start] - position[end]))
np.save('/apdcephfs/private_zaixizhang/exp_gen/target_dist.npy', bond_dist)