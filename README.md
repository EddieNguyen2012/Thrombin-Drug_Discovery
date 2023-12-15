## CS 123A Project, Fall 2023
# Title: Comprehensive Quantitative Structure-Activity Relationship model utilizing Machine Learning to predict the bioactivity of molecular inhibitors against Thrombin
#### By: Manh Tuong Nguyen, Ananya Gupta and Shwethal Trikkanad

Our project aims to develop a Quantitative Structure-Activity Relationship (QSAR) model using machine learning algorithms to predict the bioactivity of molecular inhibitors against Thrombin (our target protein). The bioactivity of the potential drug would be predicted using Lipinski descriptors.

The goal of this project is to use data from the ChEMBL Database to train a model that can predict the activity of a potential drug based on its chemical properties. This can be used to study the chemical features that are critical in drug effectiveness and to study the best molecular targets to act upon a specific disease.

## Installation

This project uses **Python 3.9** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [chembl_webresource_client](https://github.com/chembl/chembl_webresource_client)
- [scikit-learn](http://scikit-learn.org/stable/)
- [rdkit](https://www.rdkit.org/)
- [lazypredict](https://lazypredict.readthedocs.io/en/latest/)
- [PaDEL](http://www.yapcwsoft.com/dd/padeldescriptor/)

The [Pycharm](https://www.jetbrains.com/pycharm/) IDE was used to collaborate and write code. The [Anaconda](https://www.anaconda.com/download/) distribution of Python was used, which already has the above packages and more included.

The PaDEL package was download and unzipped inside the main program. For running, we have a shell script padel.sh
```
$ java -Xms1G -Xmx1G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv
```
the code call the java runtime environment to compile and run.
## 1.  Extract bioactivity data for target protein from ChEMBL

- Programmatically search ChEMBL database for our target of interest.
- Isolate bioactivity data of thrombin target with ChEMBL ID `CHEMBL204` that are reported as IC50 values in nM (nanomolar) unit.
- Drop null and duplicate rows for `canonical_smiles`.
- Assign `active`, `inactive`, or `intermediate` class to molecules based on IC50 values.

The final obtained dataframe is saved as `bioactivity_curated.csv`. It has the following columns:  
1. `molecule_chembl_id`: Unique ChEMBL ID of the molecule.
2. `canonical_smiles`: Information about the chemical and molecular structure.
3. `standard_value`: IC50 values.
4. `class`: IC50 values binned into `active`, `inactive`, and `intermediate`.

For the sake of this project, we will only work on active and inactive compounds.

## 2.  Data Preprocessing and Exploratory Data Analysis

- Use `rdkit` to compute Lipinski descriptors for the compounds compiled in the dataset from the previous step.
- Clean up SMILES data to keep the longest strand and update the `canonical_smiles` column.
- Calculate Lipinski descriptors for each compound using the `lipinski` function.
- Convert IC50 values to pIC50 values to allow for easier comparison of bioactivity across compounds with wide-ranging IC50 values.

Threshold values pIC50 > 6 = Actives and pIC50 < 5 = Inactives were used to define actives and inactives  

- Remove compounds of intermediate class from the analysis, focusing on 'active' and 'inactive' compounds only.
- Save the processed dataset to `Final_with_pIC50.csv` for subsequent analysis.

#### Statistical Analysis

- Perform the Mann-Whitney U test using `mannwhitney` function  to compare the distribution of Lipinski's descriptors between active and inactive compounds.

Theis is used to calculate the U statistic and p-value for descriptors.

- Generate visualizations, including boxplots and scatter plots, to visually assess the distributions and relationships between descriptors and bioactivity classes.

- Conclude on significant differences  
## 3. Dataset Preparation and Model Building

- Extract `canonical_smiles` and `molecule_chembl_id` from the `df_2class` DataFrame and save them to a `.smi` file for descriptor calculation using the `PaDEL-Descriptor` software.

- !IMPORTANT: Calculate molecular descriptors with `padel.sh` using 
``` 
$ chmod padel.sh +rwx 
$ sh padel.sh
```
then saving the results to `descriptors_output.csv`.

- Prepare the X (independent variables, i.e., descriptors) and Y (dependent variable, i.e., pIC50 values) for regression model training.

- Combine X and Y datasets into `Training_data.csv`.

- Remove features with low variance to refine the dataset for model training.

- Split the data into training and test sets in an 80:20 ratio.

- Train a Random Forest Regression model with 100 estimators on the training set.

- Evaluate the model's performance on the test set and record the R-squared (R²) value to measure the prediction accuracy.

- Plot and comparing the experimental versus predicted pIC50 values, providing a visual assessment of the model's predictive power.

- Employ `LazyRegressor` to compare various models and predict on both training and test sets.

- Visualize the R-Squared values of different models in a bar plot and save the performance metrics to `predictions_train.csv` and `predictions_test.csv`.

- Save the model comparison plot to `predictions_train.png`.

These steps culminate in the construction of a Random Forest model, alongside a comparison of various regression models to predict the bioactivity of molecular inhibitors against Thrombin, offering insights into model performance and guiding future model selection.

- The accuracy of the model was compared to other regressor models and reported with statistical measures such as R-square, adjusted R-Square, RMSE and Time Taken by each model

## Future Work

- Optimize the Random Forest model parameters to enhance prediction accuracy.
- Test the model on an external validation set for further assessment of its predictive capabilities.
- Explore additional machine learning algorithms and compare their performance.
- Integrate the trained model into a drug discovery pipeline to aid in the identification of new Thrombin inhibitors.


