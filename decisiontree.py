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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


# PREPROCESSING
# Load dataset from CSV file
df = pd.read_csv('C:/Users/user/OneDrive/Desktop/Datathon/ARASANFORPRESIDENT/train_person.csv')
df2 = pd.read_csv('C:/Users/user/OneDrive/Desktop/Datathon/ARASANFORPRESIDENT/train_vehicle.csv')

combinedf = pd.merge(df, df2, on = "CASENUM", how = "outer")

# Drop unnecessary columns (modify as needed)
combinedf = combinedf.drop(columns=["VIN", "C_ID_NO", "REGION_x", "VEHNO_x", "PJ_x", "CASENUM", "PERNO", "PSU_x", "AGE", "SEX", "EJECT", "VEH_ALCH", "NUM_INJV", "MODEL_YR", "VIOLATN", "MAX_VSEV", "HIT_RUN", "P_CRASH1", "IMPACT", "VEH_ROLE", "BODY_TYP", "MHENUM", "FACTOR", "DRMAN_AV", "DR_DSTRD","AXLES","OCC_INVL","VROLE_I","VLTN_I"])
 
# NOTE : NO MISSING VALUES TO FILL
# Features is represented by numbers already

#Deciding x and y axis
X = combinedf.drop(columns=["INJSEV_H"])
y = combinedf["INJSEV_H"]



# SCALLING

#for null statement
combinedf["PERALC_H"] = combinedf["PERALC_H"].replace({8: np.nan, 9: np.nan})
combinedf["HOSPITAL"] = combinedf["HOSPITAL"].replace({9: np.nan})
combinedf["PER_TYPE"] = combinedf["PER_TYPE"].replace({9: np.nan})
combinedf["PER_ALCH"] = combinedf["PER_ALCH"].replace({9: np.nan})
combinedf["REST_SYS"] = combinedf["REST_SYS"].replace({8: np.nan, 9: np.nan})
combinedf["PER_DRUG"] = combinedf["PER_DRUG"].replace({97: np.nan, 98: np.nan, 99: np.nan})
combinedf["IMPAIRMT"] = combinedf["IMPAIRMT"].replace({9: np.nan})
combinedf["SAF_EQMT"] = combinedf["SAF_EQMT"].replace({8: np.nan, 9: np.nan})
combinedf["VEH_SEV"] = combinedf["VEH_SEV"].replace({9: np.nan})
combinedf["AIRBAG"] = combinedf["AIRBAG"].replace({8: np.nan, 9: np.nan})
combinedf["PCRASH_3"] = combinedf["PCRASH_3"].replace({98: np.nan , 99: np.nan})
combinedf["PCRASH4"] = combinedf["PCRASH4"].replace({9: np.nan})
combinedf["PCRASH5"] = combinedf["PCRASH5"].replace({99: np.nan})
combinedf["ROLLOVER"] = combinedf["ROLLOVER"].replace({99: np.nan})
combinedf["CARGO_TYP"] = combinedf["CARGO_TYP"].replace({98: np.nan, 99: np.nan})
combinedf["AGE_H"] = combinedf["AGE_H"].replace({99: np.nan})
combinedf["SEAT_H"] = combinedf["SEAT_H"].replace({99: np.nan})
combinedf["LOCATN"] = combinedf["LOCATN"].replace({99: np.nan})
combinedf["ACTION"] = combinedf["ACTION"].replace({99: np.nan})
combinedf["DAM_AREA"] = combinedf["DAM_AREA"].replace({9: np.nan})
combinedf["SPEC_USE"] = combinedf["SPEC_USE"].replace({99: np.nan})
combinedf["EMCY_USE"] = combinedf["EMCY_USE"].replace({9: np.nan})
combinedf["TOWED"] = combinedf["TOWED"].replace({9: np.nan})
combinedf["DR_PRES"] = combinedf["DR_PRES"].replace({9: np.nan})
combinedf["SPEEDREL"] = combinedf["SPEEDREL"].replace({9: np.nan})
combinedf["IMPACT_H"] = combinedf["IMPACT_H"].replace({99: np.nan})



ordinal_features = ["INJSEV_H","EJECT_I","VEH_SEV", "PCRASH_3", "PCRASH4", "PCRASH5", "ROLLOVER","CARGO_TYP","MXVSEV_I"]
# count ( there is relationship)

categorial_features = ["SEX_H","PERALC_H","PER_TYPE","HOSPITAL","PER_ALCH","REST_SYS","PER_DRUG","IMPAIRMT","SAF_EQMT","AIRBAG","ACTION","STR_VEH","PCRASH_2","DAM_AREA" , "MAKE", "BODY_TYPH", "SPEC_USE" , "JACKNIFE", "FIRE", "HITRUN_I" , "EMCY_USE", "TRAILER", "TOWED", "ACC-TYPE", "DR_PRES" , "SPEEDREL", "IMPACT_H", "V_ALCH_I", "V_EVNT_H"]
# no relationshiop ( like yes or no )

numerical_features = ["AGE_H","SEAT_POS","LOCATN","WEIGHT", "MDLYR_I", "NUMOCCS", "SPEED", "NUMINJ_I"]
# counting 


# ================================
# PIPELINES
# ================================

# Numerical pipeline (scaling with imputation for missing values)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),       # Fill missing values with mean
    ('scaler', StandardScaler())                       # Scale features to 0 mean, 1 std
])

# Ordinal pipeline (treat ordinal values as ordered integers)
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),                     # Keeps ordinal nature
    ('scaler', StandardScaler())                       # Optional: scale ordinal values too
])

# Categorical pipeline (one-hot encode)
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode into binary columns
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('ord', ordinal_pipeline, ordinal_features),
    ('cat', categorical_pipeline, categorial_features)
])


# ================================
# SPLIT & PREPROCESS DATA
# ================================


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)


# ================================
# FEATURE SELECTION + MODEL
# ================================


# Train a Random Forest to get feature importances ( train a random forest on the data X and Y ) 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Use SelectFromModel to select important features ( train random forest to identify important features ) 
selector = SelectFromModel(model, prefit=True) # for individual trees 
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train a new model with selected features ( train it again using the already trained data )
model_selected = RandomForestClassifier(random_state=42) # to combine all the individual trees 
model_selected.fit(X_train_selected, y_train)

# Evaluate the model
accuracy = model_selected.score(X_test_selected, y_test)
print(f"Accuracy with selected features: {accuracy}")
