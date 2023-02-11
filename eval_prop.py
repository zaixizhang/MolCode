import pickle
import torch
from utils import check_validity, mol_stats, stats_bond_dist
import numpy as np
from rdkit import Chem
import os
from pyscf import gto, dft
from pyscf.prop.polarizability.rhf import dipole
from scipy.constants import physical_constants
EH2EV = physical_constants['Hartree energy in eV'][0]


def geom2gap(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)'  # QM9
    mol.nelectron += mol.nelectron % 2  # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    nocc = mol.nelectron // 2
    homo = mf.mo_energy[nocc - 1] * EH2EV
    lumo = mf.mo_energy[nocc] * EH2EV
    gap = lumo - homo
    return gap


def compute_homo_lumo_gap(mol_dict, valid_list, out_path):
    ptb = Chem.GetPeriodicTable()
    gap_list = []
    keys = []
    for k in mol_dict.keys():
        keys.append(k)
    keys.sort(reverse=True)

    id = 0
    for n_atoms in keys:
        numbers, positions = mol_dict[n_atoms]['_atomic_numbers'], mol_dict[n_atoms]['_positions']
        for pos, num in zip(positions, numbers):
            if not valid_list[id]:
                id += 1
                continue
            try:
                geom = [[ptb.GetElementSymbol(int(z)), pos[i]] for i, z in enumerate(num)]
                gap = geom2gap(geom)
                print(gap)
                gap_list.append(gap)
            except:
                id += 1
            else:
                id+=1
                file_obj = open(os.path.join(out_path, 'gap_out.txt'), 'a')
                file_obj.write('{:.5f}\n'.format(gap))
                file_obj.close()

    mean, median = np.mean(gap_list), np.median(gap_list)
    best = np.min(gap_list)
    good_per = np.sum(np.array(gap_list) <= 4.5) / len(gap_list)

    return {'mean': mean, 'median': median, 'best': best, 'good_per': good_per}, gap_list


def geom2alpha(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)'  # QM9
    # mol.basis = '6-31G*' # Kddcup
    mol.nelectron += mol.nelectron % 2  # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    polar = mf.Polarizability().polarizability()
    xx, yy, zz = polar.diagonal()
    return (xx + yy + zz) / 3


def compute_alpha(mol_dict, valid_list, out_path):
    ptb = Chem.GetPeriodicTable()
    alpha_list = []
    keys = []
    for k in mol_dict.keys():
        keys.append(k)
    keys.sort(reverse=True)

    id = 0
    for n_atoms in keys:
        numbers, positions = mol_dict[n_atoms]['_atomic_numbers'], mol_dict[n_atoms]['_positions']
        for pos, num in zip(positions, numbers):
            if not valid_list[id]:
                id += 1
                continue

            geom = [[ptb.GetElementSymbol(int(z)), pos[i]] for i, z in enumerate(num)]
            try:
                alpha = geom2alpha(geom)
                print(alpha)
                alpha_list.append(alpha)
            except:
                id += 1
            else:
                id += 1
                file_obj = open(os.path.join(out_path, 'alpha_out.txt'), 'a')
                file_obj.write('{:.5f}\n'.format(alpha))
                file_obj.close()

    mean, median = np.mean(alpha_list), np.median(alpha_list)
    best = np.max(alpha_list)
    good_per = np.sum(np.array(alpha_list) >= 91) / len(alpha_list)

    return {'mean': mean, 'median': median, 'best': best, 'good_per': good_per}, alpha_list

if __name__ == '__main__':
    out_path = '/zaixizhang/exp_gen/'
    mol_dict = pickle.load(open(out_path+'0.50_0.70_0.50_0.70_0.50.mol_dict', 'rb'))
    results, valid_list, con_mat_list = check_validity(mol_dict, out_path)
    bond_dist = bond_stats(mol_dict, valid_list, con_mat_list)
    np.save(out_path + 'angle_gen.npy', bond_dist)
    #result, prop_list = compute_alpha(mol_dict, valid_list, out_path)
    #result, prop_list = compute_homo_lumo_gap(mol_dict, valid_list, out_path)
