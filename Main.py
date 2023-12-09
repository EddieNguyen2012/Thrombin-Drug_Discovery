import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from chembl_webresource_client.new_client import new_client

# Target search for thrombin
target = new_client.target
target_query = target.search('thrombin')
targets = pd.DataFrame.from_dict(target_query)

selected_target = targets.target_chembl_id[1]

# activity = new_client.activity
# res = activity.filter(target_chembl_id=	selected_target).filter(standard_type="IC50")

# the original number of rows in raw data set is 3421
with open("thrombin_01_bioactivity_data_raw.csv", "r") as data:
    df = pd.DataFrame.from_dict(pd.read_csv(data))

df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]

df2_nr = df2.drop_duplicates(['canonical_smiles'])

# After clearing up missing data rows, we have 3343

df2_nr.to_csv('fin_thrombin_01_bioactivity_data_raw_na_values_removed.csv', index=False)

selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']  # standard_value = IC50
df3 = df2_nr[selection]


def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:

        # rule of 5 of Lipinski
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])

        if (i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors


# extract lipinski descriptor
df_lipinski = lipinski(df3.canonical_smiles)

df_combined = pd.concat([df3, df_lipinski], axis=1)

df_combined.to_csv("df_combined", index=False)  # df_combine rows = ['molecule_chembl_id', 'canonical_smiles',
# 'standard_value', "MW", "LogP", "NumHDonors", "NumHAcceptors"]


