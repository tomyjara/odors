import csv
from rdkit import Chem

# Define your SMILES strings
molecules = {
    'citral1': 'CC(=CCC/C(=C/C=O)/C)C',
    'citral0': 'CC(=CCCC(=CC=O)C)C',
    'caproate': 'CCCCCC(=O)OCC=C',
    'fructone': 'CCOC(=O)CC1(OCCO1)C',
    'linalool': 'CC(=CCCC(C)(C=C)O)C',
    'vanillin': 'COC1=C(C=CC(=C1)C=O)O',
    'acetophenone': 'CC(=O)C1=CC=CC=C1',
    'hexanal': 'CCCCCC=O',
    'alphapinene': 'CC1=CCC2CC1C2(C)C',
    'cyclo': 'CC1(CSC(CS1)(C)O)O',  # cyclodithalfarol
    'pentenoic': 'C=CCCC(=O)O'  # 4-pentenoic acid
}

# Convert SMILES to InChI and write to CSV
with open('dataset/common_tags/molecules_inchi.csv', 'w', newline='') as csvfile:
    fieldnames = ['common_name', 'smiles', 'inchi']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule was correctly parsed
            inchi = Chem.MolToInchi(mol)
        else:
            inchi = "Invalid SMILES"
        writer.writerow({'name': name, 'smiles': smiles, 'inchi': inchi})

print("InChI strings have been saved to 'molecules_inchi.csv'")
