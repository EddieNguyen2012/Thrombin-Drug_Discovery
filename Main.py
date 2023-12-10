
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from chembl_webresource_client.new_client import new_client
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

#Function definition
def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i * (10 ** -9)  # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', axis=1)

    return x

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


def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', axis=1)

    return x
#######################################





#Main program

# Target search for thrombin
target = new_client.target
target_query = target.search('thrombin')
targets = pd.DataFrame.from_dict(target_query)

# from above search find target of interest
# print(targets[["organism","target_chembl_id","target_type"]])

# isolate Id CHEMBL204 in selected_target var
selected_target = targets.target_chembl_id[1]

# isolate bioactivity for target based on std type
# activity = new_client.activity
# res = activity.filter(target_chembl_id=	selected_target).filter(standard_type="IC50")
#

# view res in dataframe
# res converted to dataframe everytime is computationally tedious
# we hence save it as a csv file and view the resulting dataframe by opening input file

with open("thrombin_01_bioactivity_data_raw.csv", "r") as data:
    df = pd.DataFrame.from_dict(pd.read_csv(data))


print('Unfiltered dataframe has {len(df)} rows')
# df has 3420 rows, 46 columns

# we check if df has any null values for standard_value and canonical_smiles
print('Total rows of null values in standard_value column of df',df.standard_value.isnull().sum())
# 106 rows
print('Total rows of null values in canonical_smiles column of df',df.canonical_smiles.isnull().sum())
# 0 rows


# drop rows with null values for standard_value and canonical_smiles
df2 = df.dropna(subset=['standard_value', 'canonical_smiles'])
print('Size after dropping null values', len(df2))

# check for unique canonical_smiles and drop duplicates and save to csv
print('Number of unique canonical_smiles', df2.canonical_smiles.nunique())
df2_nr = df2.drop_duplicates(['canonical_smiles'])

# df2_nr.to_csv('fin_thrombin_01_bioactivity_data_raw_na_values_removed.csv', index=False)

# remove unnecessary columns from the
selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']  # standard_type = IC50
df3 = df2_nr[selection]


bioactivity_threshold = []
for i in df3.standard_value:
    if float(i) >= 10000:
        bioactivity_threshold.append("inactive")
    elif float(i) <= 1000:
        bioactivity_threshold.append("active")
    else:
        bioactivity_threshold.append("intermediate")

bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df4 = pd.concat([df3, bioactivity_class], axis=1)

df4.to_csv("bioactivity_curated.csv")

df_no_smiles = df4.drop(columns='canonical_smiles')

smiles = []

for i in df.canonical_smiles.tolist():
    cpd = str(i).split('.')
    cpd_longest = max(cpd, key=len)
    smiles.append(cpd_longest)

smiles = pd.Series(smiles, name='canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles, smiles], axis=1)

# extract lipinski descriptor
df_lipinski = lipinski(df_clean_smiles.canonical_smiles)

df_combined = pd.concat([df4, df_lipinski], axis=1)

df_combined.to_csv("df_combined.csv", index=False)  # df_combine rows = ['molecule_chembl_id', 'canonical_smiles',
# 'standard_value', "MW", "LogP", "NumHDonors", "NumHAcceptors"]

df_norm = norm_value(df_combined)


df_final = pIC50(df_norm)

#remove intermediate candidates
df_2class = df_final[df_final['class'] != 'intermediate']

df_2class.to_csv('Final_with_pIC50.csv')

#Prepare data for descriptor calculation
selection = ['canonical_smiles','molecule_chembl_id']

df_selection = df_2class[selection]

df_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

# run descriptor calculation by using padel.sh shell script
# chmod +rwx padel.sh
# sh padel.sh
# Data saved to descriptor_output.csv

#Prepare X and Y for regression model training
df3_X = pd.read_csv('descriptors_output.csv').drop(columns=['Name'])
df3_Y = df_2class['pIC50']

#combine X and Y
dataset3 = pd.concat([df3_X,df3_Y], axis=1)

dataset3.to_csv('Training_data.csv', index=False)

#Model building
X = dataset3.drop('pIC50', axis=1) #X.shape = (3330, 881)
Y = dataset3.pIC50 # Y.shape = (3330,)

#Drop NaN values
X.dropna(axis=0, inplace=True)
Y.dropna(axis=0, inplace=True)

# Remove low variance features
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = selection.fit_transform(X) # X.shape = (3330, 154)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train.shape, Y_train.shape = ((2664, 154), (2664,))
# X_test.shape, Y_test.shape = ((666, 154), (666,))

# Building Regression Model using Random Forest
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
r2 = model.score(X_test, Y_test)

Y_pred = model.predict(X_test)

# Scatter Plot of Experimental vs Predicted pIC50 Values
sns.set(color_codes=True)
sns.set_style("white")

ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)

plt.show()