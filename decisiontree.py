import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# === Load Datasets ===
df = pd.read_csv('C:/Users/pnglo/OneDrive - University of Bristol/Desktop/University/train/ARASANFORPRESIDENT/hidden_person.csv')
df2 = pd.read_csv('C:/Users/pnglo/OneDrive - University of Bristol/Desktop/University/train/ARASANFORPRESIDENT/test_vehicle.csv')

# === Merge Datasets ===
combinedf = pd.merge(df, df2, on="CASENUM", how="outer")

# === Check MXVSEV_I exists before proceeding
if "MXVSEV_I" not in combinedf.columns:
    raise ValueError("MXVSEV_I not found in merged dataframe. Cannot derive INJSEV_H.")

# === Create INJSEV_H from MXVSEV_I ===
def derive_injury_severity(row):
    if pd.isna(row['MXVSEV_I']):
        return np.nan
    elif row['MXVSEV_I'] == 0:
        return 0
    elif row['MXVSEV_I'] == 1:
        return 1
    elif row['MXVSEV_I'] == 2:
        return 2
    else:
        return 3

combinedf["INJSEV_H"] = combinedf.apply(derive_injury_severity, axis=1)

# ✅ At this point, INJSEV_H has been created
# You can now safely use it as your target column


# === Drop unnecessary columns ===
# ONLY drop after extracting target
y = combinedf["INJSEV_H"]  # Save target variable
X = combinedf.drop(columns=["INJSEV_H", "VIN", "C_ID_NO", "REGION_x", "VEHNO_x", "PJ_x", "CASENUM", "PERNO", "PSU_x",
                            "AGE", "SEX", "EJECT", "VEH_ALCH", "NUM_INJV", "MODEL_YR", "VIOLATN", "MAX_VSEV", "HIT_RUN",
                            "P_CRASH1", "IMPACT", "VEH_ROLE", "BODY_TYP", "MHENUM", "FACTOR", "DRMAN_AV", "DR_DSTRD",
                            "AXLES", "OCC_INVL", "VROLE_I", "VLTN_I"], errors='ignore')  # errors='ignore' skips missing cols

# === Replace "null" values (encoded as numbers) with np.nan ===
null_map = {
    "PERALC_H": [8, 9], "HOSPITAL": [9], "PER_TYPE": [9], "PER_ALCH": [9],
    "REST_SYS": [8, 9], "PER_DRUG": [97, 98, 99], "IMPAIRMT": [9],
    "SAF_EQMT": [8, 9], "VEH_SEV": [9], "AIRBAG": [8, 9],
    "PCRASH_3": [98, 99], "PCRASH4": [9], "PCRASH5": [99],
    "ROLLOVER": [99], "CARGO_TYP": [98, 99], "AGE_H": [99], "SEAT_H": [99],
    "LOCATN": [99], "ACTION": [99], "DAM_AREA": [9], "SPEC_USE": [99],
    "EMCY_USE": [9], "TOWED": [9], "DR_PRES": [9], "SPEEDREL": [9],
    "IMPACT_H": [99]
}

for col, values in null_map.items():
    if col in X.columns:
        X[col] = X[col].replace(dict.fromkeys(values, np.nan))

# === Column categorization ===
ordinal_features = ["EJECT_I", "VEH_SEV", "PCRASH_3", "PCRASH4", "PCRASH5", "ROLLOVER", "CARGO_TYP", "MXVSEV_I"]
categorical_features = ["SEX_H", "PERALC_H", "PER_TYPE", "HOSPITAL", "PER_ALCH", "REST_SYS", "PER_DRUG", "IMPAIRMT",
                        "SAF_EQMT", "AIRBAG", "ACTION", "STR_VEH", "PCRASH_2", "DAM_AREA", "MAKE", "BODY_TYPH",
                        "SPEC_USE", "JACKNIFE", "FIRE", "HITRUN_I", "EMCY_USE", "TRAILER", "TOWED", "ACC-TYPE",
                        "DR_PRES", "SPEEDREL", "IMPACT_H", "V_ALCH_I", "V_EVNT_H"]
numerical_features = ["AGE_H", "SEAT_POS", "LOCATN", "WEIGHT", "MDLYR_I", "NUMOCCS", "SPEED", "NUMINJ_I"]

# === Pipelines ===
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, [col for col in numerical_features if col in X.columns]),
    ('ord', ordinal_pipeline, [col for col in ordinal_features if col in X.columns]),
    ('cat', categorical_pipeline, [col for col in categorical_features if col in X.columns])
])

# === Train-test split ===
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Apply preprocessing ===
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# === Train model for feature selection ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Feature selection ===
selector = SelectFromModel(model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# === Final model ===
model_selected = RandomForestClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)

# === Evaluate ===
accuracy = model_selected.score(X_test_selected, y_test)
print(f"Accuracy with selected features: {accuracy:.4f}")
