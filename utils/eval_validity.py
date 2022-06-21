from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import networkx as nx
import collections
from collections import defaultdict
import itertools
import copy
from scipy.spatial import distance_matrix
from rdkit.Chem.rdchem import BondType
import os
import pandas as pd
global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[6] = [4]
atomic_valence[7] = [3]
atomic_valence[8] = [2]
atomic_valence[9] = [1]

atomic_valid_bond = defaultdict(list)
atomic_valid_bond[(1, 6)] = 1.1284
atomic_valid_bond[(1, 7)] = 1.0478
atomic_valid_bond[(1, 8)] = 1.0187
atomic_valid_bond[(6, 6)] = 1.7721
atomic_valid_bond[(6, 7)] = 1.7876
atomic_valid_bond[(6, 8)] = 1.5731
atomic_valid_bond[(6, 9)] = 1.3620
atomic_valid_bond[(7, 7)] = 1.4208
atomic_valid_bond[(7, 8)] = 1.7692


def get_AC(mol, xyz):
    """

    Generate adjacent matrix from atoms and coordinates.
    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not
    covalent_factor - 1.3 is an arbitrary factor
    args:
        mol - rdkit molobj with 3D conformer
    optional
        covalent_factor - increase covalent bond length threshold with facto
    returns:
        AC - adjacent matrix
    """
    # Calculate distance matrix
    # dMat = rdkit.Chem.Get3DDistanceMatrix(mol)
    dMat = distance_matrix(xyz, xyz)
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(1, num_atoms):
        a_i = mol.GetAtomWithIdx(i).GetAtomicNum()
        for j in range(i):
            a_j = mol.GetAtomWithIdx(j).GetAtomicNum()
            if (min(a_i, a_j), max(a_i, a_j)) in atomic_valid_bond:
                threshold = atomic_valid_bond[(min(a_i, a_j), max(a_i, a_j))]
                if dMat[i, j] <= threshold and AC[j].sum() < max(atomic_valence[a_j]) and AC[i].sum() < max(atomic_valence[a_i]):
                    AC[i, j] = 1
                    AC[j, i] = 1

    return AC


def get_proto_mol(atoms):
    """
    """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(int(atoms[i]))
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def xyz2AC_vdW(atoms, xyz):
    # Get mol template
    mol = get_proto_mol(atoms)
    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2])))
    mol.AddConformer(conf)
    AC = get_AC(mol, xyz)

    return AC, mol


def get_bonds(UA, AC):
    """

    """
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1:]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, use_graph=True):
    """

    """

    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def get_UA(maxValence_list, valence_list):
    """
    """
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """
    """
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    """
    """
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def BO_is_OK(BO, AC, DU, valences):
    """
    Sanity of bond-orders

    args:
        BO -
        AC -
        charge -
        DU -


    optional
        allow_charges_fragments -


    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)

    if check_sum:
        return True

    return False


def check_connected(AC):
    seen, queue = {0}, collections.deque([0])
    while queue:
        vertex = queue.popleft()
        for node in np.argwhere(AC[vertex] > 0).flatten():
            if node not in seen:
                seen.add(node)
                queue.append(node)
    # if the seen nodes do not include all nodes, there are disconnected
    #  parts and the molecule is invalid
    if seen != {*range(len(AC))}:
        return False
    return True


def AC2BO(AC, atoms, use_graph=True):
    """

    implemenation of algorithm shown in Figure 2

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure

    """
    if not check_connected(AC):
        return AC, 0

    global atomic_valence

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))

    for _, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        # valence can't be smaller than number of neighbourgs
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
        if not possible_valence:
            return AC, 0
        valences_list_of_lists.append(possible_valence)

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    for valences in valences_list:

        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = (len(UA) == 0)
        if check_len:
            check_bo = BO_is_OK(AC, AC, DU_from_AC, valences)
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, 1

        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(BO, AC, DU_from_AC, valences)

            if status:
                return BO, 1
            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences):
                best_BO = BO.copy()

    return best_BO, 1


def BO2mol(mol, BO_matrix, atoms):
    """
    based on code written by Paolo Toscani

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity

    """

    l = len(BO_matrix)
    l2 = len(atoms)

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and Atoms {1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    return mol


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`bool`, True if chemically valid, False otherwise
    """

    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`bool`, True if no valency issues, False otherwise
    """

    try:
        s = Chem.MolToSmiles(mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except:
        return False


def xyz2mol(atoms, coordinates, use_graph=True):
    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC_vdW(atoms, coordinates)

    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    BO, valid = AC2BO(AC, atoms, use_graph)
    mol = BO2mol(mol, BO, atoms)
    return BO, valid, mol


def check_validity(mol_dicts, path):
    n_generated, n_valid = 0, 0,
    valid_list, con_mat_list = [], []
    smiles_list = []

    for n_atoms in mol_dicts:
        numbers, positions = mol_dicts[n_atoms]['_atomic_numbers'], mol_dicts[n_atoms]['_positions']

        for _, (pos, num) in enumerate(zip(positions, numbers)):
            con_mat, valid, mol = xyz2mol(num, pos)
            n_generated += 1
            mol = Chem.RemoveHs(mol)
            smile = Chem.MolToSmiles(mol)
            smiles_list.append(smile)
            if valid:
                n_valid += 1
                valid_list.append(1)
                #Chem.MolToMolFile(mol, path+'saved_mol/{:}.mol'.format(n_valid))
            else:
                valid_list.append(0)
            con_mat_list.append(con_mat)

    file_obj = open(os.path.join('/apdcephfs/private_zaixizhang/exp_gen/7/', 'grid_search.txt'), 'a')
    file_obj.write('n_generated {:.2f} valid_ratio {:.2f} uniqueness {:.2f} \n'.format(n_generated, n_valid / n_generated, len(set(smiles_list))/n_generated))
    file_obj.close()

    return {'n_generated': n_generated, 'valid_ratio': n_valid / n_generated, 'uniqueness':len(set(smiles_list))/n_generated}, valid_list, con_mat_list

def mol_stats(mol_dicts):
    n_generated, n_valid = 0, 0
    bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
    rings=[]
    bonds=[]
    atoms=[]
    r3,r4,r5,r6=0.,0.,0.,0.
    b1,b2,b3=0.,0.,0.
    c,h,o,n,f=0.,0.,0.,0.,0.

    for n_atoms in mol_dicts:
        if n_atoms<11:
            continue
        numbers, positions = mol_dicts[n_atoms]['_atomic_numbers'], mol_dicts[n_atoms]['_positions']

        for _, (pos, num) in enumerate(zip(positions, numbers)):
            con_mat, valid, mol = xyz2mol(num, pos)
            n_generated += 1
            ssr = Chem.GetSymmSSSR(mol)
            rings.extend([len(ssr[i]) for i in range(len(ssr))])
            bonds.extend([bond_to_type[bond.GetBondType()] for bond in mol.GetBonds()])
            atoms.extend([atom.GetAtomicNum() for atom in mol.GetAtoms()])

    for r in rings:
        if r==3:
            r3+=1
        if r==4:
            r4+=1
        if r==5:
            r5+=1
        if r==6:
            r6+=1
    for b in bonds:
        if b==1:
            b1+=1
        if b==2:
            b2+=1
        if b==3:
            b3+=1
    for a in atoms:
        if a==6:
            c+=1
        if a==7:
            n+=1
        if a==8:
            o+=1
        if a==9:
            f+=1
        if a==1:
            h+=1
    stats={'c':c/n_generated,'n':n/n_generated,'o':o/n_generated, 'f':f/n_generated, 'h':h/n_generated, 'b1':b1/n_generated,'b2':b2/n_generated, 'b3':b3/n_generated,
           'r3':r3/n_generated,'r4':r4/n_generated,'r5':r5/n_generated,'r6':r6/n_generated}
    print(stats)

    return stats


def bond_stats(mol_dicts):
    angle_dist = {}
    id = 0

    for n_atoms in mol_dicts:
        if n_atoms<=5:
            continue
        numbers, positions = mol_dicts[n_atoms]['_atomic_numbers'], mol_dicts[n_atoms]['_positions']
        for _, (position, atom_type) in enumerate(zip(positions, numbers)):
            con_mat, valid, mol = xyz2mol(atom_type, position)
            if not valid:
                continue
            atom_ids_1, atom_ids_2 = np.nonzero(con_mat)
            bond_tmp = []
            for id1, id2 in zip(atom_ids_1, atom_ids_2):
                if id1 < id2:
                    continue
                bond_tmp.append([id1, id2])

            for i in range(len(bond_tmp) - 1):
                for j in range(i + 1, len(bond_tmp)):
                    if set(bond_tmp[i]) & set(bond_tmp[j]) != set():
                        pivotal = list(set(bond_tmp[i]) & set(bond_tmp[j]))[0]
                        p1 = list(set(bond_tmp[i]) - set({pivotal}))[0]
                        p2 = list(set(bond_tmp[j]) - set({pivotal}))[0]
                        z1, z2 = atom_type[p1], atom_type[p2]
                        z1, z2 = min(z1, z2), max(z1, z2)
                        if not (z1, z2, atom_type[pivotal]) in angle_dist:
                            angle_dist[(z1, z2, atom_type[pivotal])] = []
                        vec1 = position[p1] - position[pivotal]
                        vec2 = position[p2] - position[pivotal]
                        product = np.dot(np.array(vec1), np.array(vec2)) / (
                                    np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        angle_dist[(z1, z2, atom_type[pivotal])].append(np.arccos(product) / np.pi * 180)
                '''
                atomic1, atomic2 = num[id1], num[id2]
                z1, z2 = min(atomic1, atomic2), max(atomic1, atomic2)
                bond_type = con_mat[id1, id2]

                if not (z1, z2, bond_type) in bond_dist:
                    bond_dist[(z1, z2, bond_type)] = []
                bond_dist[(z1, z2, bond_type)].append(np.linalg.norm(pos[id1] - pos[id2]))
            id += 1'''

    return angle_dist