import pandas as pd
from matplotlib import pyplot
from pandas import read_csv, set_option
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
#%matplotlib inline
input_file = "sonar.all-data"


# comma delimited is the default
df = pd.read_csv(input_file, header = None)

set_option('display.width', 100)
df.head(20)
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
pyplot.show()
#target_class = Dataset[df.columns[len(Dataset.axes[1])-1:]]
# remove the non-numeric columns
feature = df.select_dtypes(include=['float64']);
lable = df.select_dtypes(exclude=['float64']);


#df.drop(df.columns[[60]], axis=1)  # df.columns is zero-based pd.Index
