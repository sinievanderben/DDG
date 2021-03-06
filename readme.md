# Ship ML Experiment DDG
## _Predict Gross Tonnage of the biggest cruiseships_

This experiment consists of two parts: a Jupyter Notebook and a collection of Python files. The notebook is the place where all thoughts are expressed and it shows the build-up of the final product, which is visible in the collection of Python files.  

All files can be run in the following environment called 'ddg':
```sh
conda env create -f environment.yml
```

Three models are compared: Linear regression, Random Forest and Gradient Boosting. The latter two are ensembles of models. 

## Jupyter Notebook: ship.ipynb
I think the notebook speaks for itself. It consists of the following parts:

1. Explore the dataset and prepare the data 
2. Create new features 
3. Separate the data into train, validation and test set
4. Test feature importance 
5. Find correct algorithm and Hyperparameter tuning
7. Test the algorithm 

Thoughts and reasons behind choices are explained briefly but can be further discussed during the demo. 

## Collection of Python files 

The Python files will be executed through one main.py file. In the command line you can paste the following:

```sh
python main.py
``` 

This will print out the train, validation and test error of the final used model and a special bonus.

The collection of files are:
- main.py : runs it all, rules it all
- helperfunctions.py: some messy functions which I wanted to separate but did not take enough space for own files. Contains function to split the dataset and to show the reslts 
- tuning.py performs the tuning of the RF an the GB models. 
- cleanupfeatures.py: file that contains the functions to clean up the features and perform feature selection. 
- bonus.py: comparison of a simple equation to model performance using RMSE

Extensive feature search is not done in the Python files. These files only use the results of this process. The extensive feature search is clearly explained in the Jupyter Notebook and I am happy to explain it during the demo.  


