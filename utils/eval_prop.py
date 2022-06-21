import numpy as np
from pyscf import gto, dft
from pyscf.prop.polarizability.rhf import dipole
from rdkit import Chem
from scipy.constants import physical_constants
EH2EV = physical_constants['Hartree energy in eV'][0]


def geom2gap(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)' #QM9
    mol.nelectron += mol.nelectron % 2 # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    nocc = mol.nelectron // 2
    homo = mf.mo_energy[nocc - 1] * EH2EV
    lumo = mf.mo_energy[nocc] * EH2EV
    gap = lumo - homo
    return gap


def compute_homo_lumo_gap(mol_dict, valid_list):
    ptb = Chem.GetPeriodicTable()
    gap_list = []

    id = 0
    for n_atoms in mol_dict:
        numbers, positions = mol_dict[n_atoms]['_atomic_numbers'], mol_dict[n_atoms]['_positions']
        for pos, num in zip(positions, numbers):
            if not valid_list[id]:
                id += 1
                continue

            geom = [[ptb.GetElementSymbol(int(z)), pos[i]] for i, z in enumerate(num)]
            gap = geom2gap(geom)
            gap_list.append(gap)
            id += 1
    
    mean, median = np.mean(gap_list), np.median(gap_list)
    best = np.min(gap_list)
    good_per = np.sum(np.array(gap_list) <= 4.5) / len(gap_list)

    return {'mean': mean, 'median': median, 'best': best, 'good_per': good_per}, gap_list


def geom2alpha(geom):
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = '6-31G(2df,p)' #QM9
    # mol.basis = '6-31G*' # Kddcup
    mol.nelectron += mol.nelectron % 2 # Make it full shell? Otherwise the density matrix will be 3D
    mol.build(0, 0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    polar = mf.Polarizability().polarizability()
    xx, yy, zz = polar.diagonal()
    return (xx + yy + zz) / 3


def compute_alpha(mol_dict, valid_list):
    ptb = Chem.GetPeriodicTable()
    alpha_list = []
    
    id = 0
    for n_atoms in mol_dict:
        numbers, positions = mol_dict[n_atoms]['_atomic_numbers'], mol_dict[n_atoms]['_positions']
        for pos, num in zip(positions, numbers):
            if not valid_list[id]:
                id += 1
                continue

            geom = [[ptb.GetElementSymbol(int(z)), pos[i]] for i, z in enumerate(num)]
            alpha = geom2alpha(geom)
            alpha_list.append(alpha)
            id += 1
    
    mean, median = np.mean(alpha_list), np.median(alpha_list)
    best = np.max(alpha_list)
    good_per = np.sum(np.array(alpha_list) >= 91) / len(alpha_list)

    return {'mean': mean, 'median': median, 'best': best, 'good_per': good_per}, alpha_list