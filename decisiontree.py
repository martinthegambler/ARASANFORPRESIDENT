import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset from CSV file
df = pd.read_csv("ARASANFORPRESIDENT/train_vehicle.csv")
df2 = pd.read_csv("ARASANFORPRESIDENT/train_person.csv")

combinedf = pd.merge(df, df2, on = "CASENUM", how = "outer")

# Show first 5 rows  
print(combinedf.head())    