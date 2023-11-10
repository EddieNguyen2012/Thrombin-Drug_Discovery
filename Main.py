import pandas as pd
from chembl_webresource_client.new_client import new_client

# Target search for thrombin
target = new_client.target
target_query = target.search('thrombin')
targets = pd.DataFrame.from_dict(target_query)

selected_target = targets.target_chembl_id[1]

#activity = new_client.activity
#res = activity.filter(target_chembl_id=	selected_target).filter(standard_type="IC50")

with open("thrombin_01_bioactivity_data_raw.csv", "r") as data:
    df = pd.DataFrame.from_dict(pd.read_csv(data))

df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]

df2_nr = df2.drop_duplicates(['canonical_smiles'])

df2_nr.to_csv('fin_thrombin_01_bioactivity_data_raw_na_values_removed.csv', index=False)

selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df3 = df2_nr[selection]

