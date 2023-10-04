import torch
import numpy as np
from rdkit import Chem
from pprint import pprint

from possible_atom_features import possible_atom_list, possible_bond_types, possible_hybridization_list, \
    possible_formal_charges, possible_num_radical_electrons, \
    possible_bond_conjugations, possible_bond_stereos

from torch_geometric.data import Data

possible_numH_list = [0, 1, 2, 3, 4]
# mismo valor no aporta nada
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_chirality_list = ['R', 'S']

min_mass = [None]
max_mass = [0]


def show_atoms_features(mols):
    atom_set = set()
    degrees = set()
    valences = set()
    hybridizations = set()
    formal_charges = set()
    num_radical_electrons = set()
    masses = set()
    chiralities = set()
    for mol in mols:
        atoms = mol.GetAtoms()
        for atom in atoms:
            atom_set.add(atom.GetSymbol())
            degrees.add(atom.GetDegree())
            valences.add(atom.GetImplicitValence())
            hybridizations.add(atom.GetHybridization())
            formal_charges.add(atom.GetFormalCharge())
            num_radical_electrons.add(atom.GetNumRadicalElectrons())
            masses.add(atom.GetMass())
            try:
                chiralities.add(atom.GetProp('_CIPCode'))
            except:
                pass

    pprint(atom_set)
    pprint(degrees)
    pprint(valences)
    pprint(hybridizations)
    pprint(formal_charges)
    pprint(chiralities)
    pprint(num_radical_electrons)
    print('max mass', max(masses))
    print('min mass', min(masses))


def atom_features(atom, explicit_H=False, use_chirality=True):
    # sweet vs bitter
    min_mass_f = 6.941
    max_mass_f = 200.59

    results = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom_list) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
              one_of_k_encoding(atom.GetHybridization(), possible_hybridization_list) + \
              one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charges) + \
              one_of_k_encoding(atom.GetNumRadicalElectrons(), possible_num_radical_electrons) + \
              [atom.GetIsAromatic(), atom.IsInRing()] + \
              [(atom.GetMass() - min_mass_f) / (max_mass_f - min_mass_f)]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), possible_numH_list)
    if use_chirality:
        try:
            # print(str(atom.GetProp('_CIPCode')))
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), possible_chirality_list) + [
                atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    # print('max mass', max(masses))
    # print('min mass', min(masses))
    return np.array(results)


def edge_attributes(mol):
    # adj = Chem.GetAdjacencyMatrix(mol)
    bonds = mol.GetBonds()
    edges = [[], []]
    features = []
    for bond in bonds:
        type = str(bond.GetBondType())
        type = one_of_k_encoding(type, possible_bond_types)

        conjugated = str(bond.GetIsConjugated())
        conjugated = one_of_k_encoding(conjugated, possible_bond_conjugations)

        stereo = str(bond.GetStereo())
        stereo = one_of_k_encoding(stereo, possible_bond_stereos)

        edge_feats = []
        edge_feats += type
        edge_feats += conjugated
        edge_feats += stereo

        features.append(edge_feats)

        edges[0] += [bond.GetBeginAtomIdx()]  # , bond.GetEndAtomIdx()]
        edges[1] += [bond.GetEndAtomIdx()]  # , bond.GetBeginAtomIdx()]
    return edges, features


def mol2graph(mol, label, name='None'):
    atoms = mol.GetAtoms()
    node_f = [atom_features(atom) for atom in atoms]
    edge_index, edge_features = edge_attributes(mol)

    if len(edge_features) == 0:
        return None

    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=torch.tensor(np.array(node_f), dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=label
                )
    data.name = name
    #print('MOL NAME', name, 'NUMBER OF ATOMS', len(atoms))

    return data


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def inchisToMols(inchis):
    mols = []
    for i, mol in enumerate(inchis):
        if mol != None:
            try:
                mols.append(Chem.rdinchi.InchiToMol(mol)[0])
            except Exception as e:
                print(e)
                print(i)
                pass
    return mols
