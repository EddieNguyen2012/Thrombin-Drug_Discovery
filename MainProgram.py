
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
from chembl_webresource_client.new_client import new_client
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import lazypredict
from lazypredict.Supervised import LazyRegressor
# Function definition
def pIC50(input):
    """ use a dataframe as df_with_IC50 and convert IC50 values from nm to M,
        apply -log 10 to it
        """
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i * (10 ** -9)  # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', axis=1)

    return x

def lipinski(smiles, verbose=False):
    """ The function takes a dataframe series as input and uses the chemical formula of each molecule
    from the canonical_smiles column to produces a dataframe including the lipinski descriptors"""

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
    """ Values greater than 100,000,000
        fixed at 100,000,000 otherwise the negative logarithmic value will become negative"""

    norm = []

    for i in input['standard_value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', axis=1)

    return x

def updated_bioactivity_class(df, pIC50_column='pIC50'):
    """
    Assigns bioactivity class based on pIC50 values.
    Actives: pIC50 > 6
    Inactives: pIC50 < 5
    Intermediates: 5 <= pIC50 <= 6
    """
    bioactivity_class = []
    for value in df[pIC50_column]:
        if value > 6:
            bioactivity_class.append("active")
        elif value < 5:
            bioactivity_class.append("inactive")
        else:
            bioactivity_class.append("intermediate")

    df['class'] = bioactivity_class
    return df



def mannwhitney(descriptor, verbose=False):
    # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/


    # actives and inactives
    selection = [descriptor, 'class']
    df = df_2class[selection]
    active = df[df['class'] == 'active']
    active = active[descriptor]

    selection = [descriptor, 'class']
    df = df_2class[selection]
    inactive = df[df['class'] == 'inactive']
    inactive = inactive[descriptor]

    # compare samples
    stat, p = mannwhitneyu(active, inactive)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))

    # interpret
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'

    results = pd.DataFrame({'Descriptor': descriptor,
                            'Statistics': stat,
                            'p': p,
                            'alpha': alpha,
                            'Interpretation': interpretation}, index=[0])
    filename = 'mannwhitneyu_' + descriptor + '.csv'
    results.to_csv(filename)

    return results

#######################################





##Main program

# Target search for thrombin
target = new_client.target
target_query = target.search('thrombin')
targets = pd.DataFrame.from_dict(target_query)

# from above search find target of interest
print(targets[["organism","target_chembl_id","target_type"]])

# isolate Id CHEMBL204 in selected_target var
thrombin_target = targets.target_chembl_id[1]

# isolate bioactivity for target based on std type
activity = new_client.activity
result_search = activity.filter(target_chembl_id=thrombin_target ).filter(standard_type="IC50")
#

# view result_search in dataframe
df_res = pd.DataFrame.from_dict(result_search)

# result_search converted to dataframe everytime is computationally tedious
# we hence save it as a csv file and view the resulting dataframe by opening df_with_IC50 file

df_res.to_csv('thrombin_01_bioactivity_data_raw.csv.csv', index = False)

with open("thrombin_01_bioactivity_data_raw.csv", "r") as data:
    df = pd.DataFrame.from_dict(pd.read_csv(data))


print('Unfiltered dataframe has shape {df.shape}')
# df has 3420 rows, 46 columns

# we check if df has any null values for standard_value and canonical_smiles
print('Total rows of null values in standard_value column of df',df.standard_value.isnull().sum())
# 106 rows
print('Total rows of null values in canonical_smiles column of df',df.canonical_smiles.isnull().sum())
# 0 rows


# drop rows with null values for standard_value and canonical_smiles
df2 = df.dropna(subset=['standard_value', 'canonical_smiles'])
print(f'Size after dropping null values, {len(df2)}')

# check for unique canonical_smiles and drop duplicates and save to csv
print('Number of unique canonical_smiles', df2.canonical_smiles.nunique())
df2_nonull = df2.drop_duplicates(['canonical_smiles'])

df2_nonull.to_csv('fin_thrombin_01_bioactivity_data_raw_na_values_removed.csv', index=False)

# remove unnecessary columns from the df
col_wanted = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']  # standard_type = IC50
df3 = df2_nonull[col_wanted]
print('Finally dataset has rows:',len(df3))


# assign bioactivity class based on IC50 unit

bioactivity_prop = []
for i in df3.standard_value:
    if float(i) >= 10000:
        bioactivity_prop.append("inactive")
    elif float(i) <= 1000:
        bioactivity_prop.append("active")
    else:
        bioactivity_prop.append("intermediate")

bioactivity_class = pd.Series(bioactivity_prop, name='class')
df4 = pd.concat([df3, bioactivity_class], axis=1)
df4 = df4.dropna().reset_index(drop=True)

df4.to_csv("bioactivity_curated.csv", index = False)

df_no_smiles = df4.drop(columns='canonical_smiles')

smiles = []

for i in df4.canonical_smiles.tolist():
    can = str(i).split('.')
    can_longest = max(can, key=len)
    smiles.append(can_longest)


smiles = pd.Series(smiles, name='canonical_smiles').dropna()

df_clean_smiles = pd.concat([df_no_smiles, smiles], axis=1)

# extract lipinski descriptor using custom function
df_lipinski = lipinski(df_clean_smiles.canonical_smiles).dropna()

df_combined = pd.concat([df4, df_lipinski], axis=1)
df_combined_cleaned = df_combined.dropna().reset_index(drop=True)

df_combined_cleaned.to_csv("df_combined.csv", index=False)  # df_combine rows = ['molecule_chembl_id', 'canonical_smiles',
# 'standard_value', "MW", "LogP", "NumHDonors", "NumHAcceptors"]

df_norm = norm_value(df_combined_cleaned)

df_final = pIC50(df_norm)
# Apply the function to your dataframe
df_final = updated_bioactivity_class(df_final)

# # UNCOMMENT TO MAKE PLOT

fig, axs = plt.subplots(2)
# visualisation of pre transformed standard_value
axs[0].hist(df_combined_cleaned['standard_value'], bins=30)
axs[0].set_title('Std_value before transformation')

# visualisation of transformed standard_value
axs[1].hist(df_final['pIC50'], bins=30)
axs[1].set_title('Std_value after transformation')
plt.tight_layout()
plt.savefig('std_val before and after transformation.png')

# remove intermediate candidates
df_2class = df_final[df_final['class'] != 'intermediate']

print(df_2class.shape) #(1985, 8)
df_2class.to_csv('Final_with_pIC50.csv')


# # Performing Chemical Space Analysis

# UNCOMMENT TO PLOT
# Plot to observe frequency of the bioactivity class

plt.figure(figsize=(5.5, 5.5))

sns.countplot(x='class', data=df_2class, edgecolor='black')
plt.title('Frequency Plot of Bioactivity Classes')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig('plot_bioactivity_class_frequency.png')

#  molecular weight (MW) and lipophilicity (LogP) of compounds was plotted.
plt.figure(figsize=(14, 5.5))

sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='class', size='pIC50', edgecolor='black', alpha=0.7,legend=True
                )
plt.title("LogP vs Molecular Weight of compounds", fontsize=14, fontweight='bold')
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
sns.regplot(x='MW', y='LogP', data=df_2class, scatter=False, color='black')
plt.axvline(x=500, linestyle='--', color='gray', linewidth=1)
plt.axhline(y=5, linestyle='--', color='gray', linewidth=1)
plt.savefig('plot_MW_vs_LogP.png')

# activity and pIC50
plt.figure(figsize=(8, 5.5))

sns.boxplot(x = 'class', y = 'pIC50', data = df_2class)
plt.title("relationship between each activity level and pIC50", fontsize=14, fontweight='bold')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.yticks(np.arange(0, df_2class['pIC50'].max() + 1, 1))
plt.savefig('plot_ic50.png')


# MW plot
plt.figure(figsize=(10, 5.5))

sns.boxplot(x='class', y='MW', data=df_2class)
plt.title("relationship between each activity level and MW", fontsize=14, fontweight='bold')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('plot_MW.png')


# logP plot
plt.figure(figsize=(10, 5.5))

sns.boxplot(x='class', y='LogP', data=df_2class)

plt.title("relationship between each activity level and LogP", fontsize=14, fontweight='bold')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')

plt.savefig('plot_LogP.png')


# h donor plot
plt.figure(figsize=(10, 5.5))

sns.boxplot(x='class', y='NumHDonors', data=df_2class)
plt.title("relationship between each activity level and HDonors", fontsize=14, fontweight='bold')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHDonors.png')


# h accept plot
plt.figure(figsize=(10, 5.5))

sns.boxplot(x='class', y='NumHAcceptors', data=df_2class)

plt.title("relationship between each activity level and HAcceptor", fontsize=14, fontweight='bold')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHAcceptors.png')

# List of descriptors to apply the Mann-Whitney test
descriptors = ['LogP', 'NumHDonors', 'NumHAcceptors', 'MW', 'pIC50']

for descriptor in descriptors:
    mannwhitney(descriptor, df_2class)




# # Prepare data for descriptor calculation
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

print(X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train.shape, Y_train.shape = ((2664, 154), (2664,))
# X_test.shape, Y_test.shape = ((666, 154), (666,))

# Building Regression Model using Random Forest
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
r2 = model.score(X_test, Y_test)

Y_pred = model.predict(X_test)

# Plot of Experimental vs Predicted pIC50 Values
sns.set(color_codes=True)
sns.set_style("white")

plt.figure(figsize=(16, 9))
ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
plt.title("Predicted Plot")
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.savefig("predicted_plot.png")

clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models_train, predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
models_test, predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

plt.figure(figsize=(16, 9))
sns.set_theme(style="whitegrid")
plt.title("Model Comparisons by R-Squared")
ax = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
ax.set(xlim=(0, 1))

predictions_train.to_csv('predictions_train.csv')
predictions_test.to_csv('predictions_test.csv')

plt.savefig('comparisons_rsquared.png')

plt.figure(figsize=(16, 9))
sns.set_theme(style="whitegrid")
plt.title("Model Comparisons by RMSE")
ax = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
ax.set(xlim=(0, 10))

plt.savefig('comparisons_rmse.png')

plt.figure(figsize=(16, 9))
sns.set_theme(style="whitegrid")
plt.title("Model Comparisons by Time taken")
ax = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
ax.set(xlim=(0, 10))
plt.savefig('comparisons_time_taken.png')