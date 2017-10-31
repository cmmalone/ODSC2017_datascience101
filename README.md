# Data Science 101 
## Katie Malone, Civis Analytics
## Open Data Science Conference, San Francisco 2017 

### Introduction
This is the git repo for the "Data Science 101" workshop at the Open Data Science Conference, Nov. 2 2017.  The aim of this workshop is to build a simple end-to-end data science modeling workflow in python, and then to take that workflow and make it generalized to repeated running on updated datasets.

### Data
The data for this workshop comes from the DrivenData.org ["Pump It Up"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) challenge and was downloaded in October 2017.  As downloaded, the data comes in large files with all data points present together, but one of the objectives of the ODSC workshop is to mock up and build a machine learning pipeline for a situation in which the data scientist will be receiving periodic data updates. In order to get a dataset that simulates this situation, there's a script (`data_setup.py`) that orders the dataset by date and then slices it into six pieces. In part 2, when we make a maintainable model, we will assume the data has already been preprocessed by this script so we can develop the pipeline on the first data slice and then add in the additional data slices. 


### Part 1: Scripted Model
The scripted model is built in an ipython notebook, `scripted_model.ipynb`.  This notebook assumes you have the training data downloaded from drivendata.org and it's stored locally in files called `well_data.csv` (features) and `well_labels.csv` (labels).  

This notebook goes through a few drafts of the steps of a modeling workflow, including transforming string categorical variables to integers (in the case of labels) or dummies (in the case of features), selecting the best features for model-building, and then building a multiclass random forest classifier for classification.


If you're following along with the workshop in San Francisco, or want to do some exercises on your own, there is a notebook called `scripted_notebook_to_fill_in.ipynb` that has cells strategically left blank that you can fill in.
### Part 2: Maintainable Model


