import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image 
import pydotplus

dataset = pd.read_csv('BaseLimpaFinal2.csv', index_col=0, header=0)
dataset = dataset.iloc[:, 2:12]
dataset.describe()

dataset['trafInac'], class_names = pd.factorize(dataset["trafInac"])

y = dataset.iloc[:, 9].values
X = dataset.drop(columns=["trafInac","gdp.pk","EIBfin","ERDPfin"]).values
features_names = list(dataset.drop(columns=["trafInac","gdp.pk","EIBfin","ERDPfin"]).columns)

#K-FOLD

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble

# Lets split the data into 5 folds. 
# We will use this 'kf'(StratiFiedKFold splitting stratergy) object as input to cross_val_score() method
# The folds are made by preserving the percentage of samples for each class.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
    
# Note that: 
# cross_val_score() parameter 'cv' will by default use StratifiedKFold spliting startergy if we just specify value of number of folds. 
# So you can bypass above step and just specify cv= 5 in cross_val_score() function
