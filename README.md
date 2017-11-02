# Data Science 101 
## Katie Malone, Civis Analytics
## Open Data Science Conference, San Francisco 2017 

### [Link to accompanying slides](https://www.dropbox.com/s/qb9jh1adaei7o9i/DataScience101.pdf?dl=0)

### Introduction
This is the git repo for the "Data Science 101" workshop at the Open Data Science Conference, Nov. 2 2017.  The aim of this workshop is to build a simple end-to-end data science modeling workflow in python, and then to take that workflow and make it generalized to repeated running on updated datasets.

### Data
The data for this workshop comes from the DrivenData.org ["Pump It Up"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) challenge and was downloaded in October 2017.  As downloaded, the data comes in large files with all data points present together, but one of the objectives of the ODSC workshop is to mock up and build a machine learning pipeline for a situation in which the data scientist will be receiving periodic data updates. In order to get a dataset that simulates this situation, there's a script (`data_setup.py`) that orders the dataset by date and then slices it into six pieces. In part 2, when we make a maintainable model, we will assume the data has already been preprocessed by this script so we can develop the pipeline on the first data slice and then add in the additional data slices. 


### Part 1: Scripted Model
The scripted model is built in an ipython notebook, `scripted_model.ipynb`.  This notebook assumes you have the training data downloaded from drivendata.org and it's stored locally in files called `well_data.csv` (features) and `well_labels.csv` (labels).  

This notebook goes through a few drafts of the steps of a modeling workflow, including transforming string categorical variables to integers (in the case of labels) or dummies (in the case of features), selecting the best features for model-building, and then building a multiclass random forest classifier for classification.


If you're following along with the workshop in San Francisco, or want to do some exercises on your own, there is a notebook called `scripted_notebook_to_fill_in.ipynb` that has cells strategically left blank that you can fill in.

### Part 2: Maintainable Model
The scripted model is great, but in the long run, that will be hard to  maintain. Say you want to re-run the notebook on different data--maybe you end up cloning the notebook so you have a record of each version of the notebook you've run (and can, for example, compare the performance of model #1 against model #2), which might be well and good but then you have to keep track of potentially many notebooks.  Moreover, suppose that later you find a bug in notebook N that originated in notebook 1--re-tracing your steps and propagating the fix can be a huge hassle.

In general, maintainable and extensible software is really tough to build in notebooks in my experience, but python scripts work nicely for making software.  The goal in this part of the workshop is to take the basic outline spelled out in the notebook (dummying, feature selection, random forest) and re-write it in a way that's broken into pieces for maintainability and extensibility.

Moreover, we're pretending now that we're in a regime where new data might be coming in periodically and you want the code to ingest and use whatever data you want to throw at it.  This is a simple idea in theory but harder in practice.  In particular, dummying the categorical features is challenging, because we'll be fitting the dummying scheme on the training data and applying it to the testing data (fitting the dummying on the entire dataset is cheating, in a sense, because when we're making predictions about future cases we will only have the training data available, and we want to practice like we play).  Making that training-to-testing dummying crosswalk ends up being technically challenging to pull off.

Anyway, in the workshop, we'll build a workflow step-by-step and the participants should be building the workflow themselves as we go.  That said, some of these steps are tricky and it would be a shame for folks to miss out on good learnin's just because they fall behind.  In that spirit, there are working pieces of code that solve each of the steps of the workflow build included in this repo; they're in the `solutions` folder.   

Pull requests are welcome.  The code in `solutions` is working code that fulfills the goal of each step but I'm sure there are better implementations out there if you want to make them.
