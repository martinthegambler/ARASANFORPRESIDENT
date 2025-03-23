import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PREPROCESSING
# Load dataset from CSV file
df = pd.read_csv("ARASANFORPRESIDENT/train_vehicle.csv")
df2 = pd.read_csv("ARASANFORPRESIDENT/train_person.csv")

combinedf = pd.merge(df, df2, on = "CASENUM", how = "outer")

# Drop unnecessary columns (modify as needed)
combinedf = combinedf.drop(columns=["VIN", "C_ID_NO", "REGION", "VEHNO", "PJ", "CASENUM", "PERNO", "PSU", "AGE", "SEX", "EJECT", "VEH_ALCH", "NUM_INJV", "MODEL_YR", "VIOLATN", "MAX_VSEV", "HIT_RUN", "P_CRASH1", "IMPACT", "VEH_ROLE", "BODY_TYP", "MHENUM", "FACTOR", "DRMAN_AV", "DR_DSTRD","AXLES","OCC_INVL","VROLE_I","VLTN_I"])

# NOTE : NO MISSING VALUES TO FILL
# Features is represented by numbers already

#Deciding x and y axis
x = combinedf.drop(columns=["INJSEV_H"])
y = combinedf("INJSEV_H")

#split the data 