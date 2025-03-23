import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# PREPROCESSING
# Load dataset from CSV file
df = pd.read_csv('C:/Users/user/OneDrive/Desktop/Datathon/ARASANFORPRESIDENT/train_person.csv')
df2 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Datathon/ARASANFORPRESIDENT/train_vehicle.csv')

combinedf = pd.merge(df, df2, on = "CASENUM", how = "outer")

# Drop unnecessary columns (modify as needed)
combinedf = combinedf.drop(columns=["VIN", "C_ID_NO", "REGION_x", "VEHNO_x", "PJ_x", "CASENUM", "PERNO", "PSU_x", "AGE", "SEX", "EJECT", "VEH_ALCH", "NUM_INJV", "MODEL_YR", "VIOLATN", "MAX_VSEV", "HIT_RUN", "P_CRASH1", "IMPACT", "VEH_ROLE", "BODY_TYP", "MHENUM", "FACTOR", "DRMAN_AV", "DR_DSTRD","AXLES","OCC_INVL","VROLE_I","VLTN_I"])
 
# NOTE : NO MISSING VALUES TO FILL
# Features is represented by numbers already

#scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Scaling
    ('model', RandomForestClassifier(random_state=42))  # Step 2: Model fitting
])

#Deciding x and y axis
X = combinedf.drop(columns=["INJSEV_H"])
y = combinedf["INJSEV_H"]

pipeline.fit(X,y)

#split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest to get feature importances
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Use SelectFromModel to select important features
selector = SelectFromModel(model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train a new model with selected features
model_selected = RandomForestClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)

# Evaluate the model
accuracy = model_selected.score(X_test_selected, y_test)
print(f"Accuracy with selected features: {accuracy}")