import os, io, zipfile, warnings, sys
warnings.filterwarnings("ignore")
print("Preparing environment...")

# install missing libs if needed
try:
    import sklearn, xgboost, lightgbm, imblearn, joblib
except Exception:
    print("Installing missing packages. This may take a minute...")
    !pip install -q xgboost lightgbm imbalanced-learn joblib openpyxl

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import joblib
from google.colab import files

print("Packages ready.")

# ---------------------------
# 1) Upload ZIP of datasets
# ---------------------------
print("\n📂 Please upload a ZIP file containing your datasets (CSV / XLSX).")
print("   After choosing file, run this cell. Colab will extract and load files automatically.")
uploaded = files.upload()  # triggers a choose-file dialog

# find the uploaded zip file
zip_path = None
for fname in uploaded.keys():
    if fname.lower().endswith(".zip"):
        zip_path = fname
        break

if zip_path is None:
    raise SystemExit("No .zip uploaded — re-run and upload thalassemia datasets zip file.")

# extract zip to 'uploaded_data' folder
extract_dir = "uploaded_data"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)
print(f"✅ Uploaded ZIP extracted to ./{extract_dir}")

# list files found
found_files = []
for root,_,files_list in os.walk(extract_dir):
    for f in files_list:
        if f.lower().endswith((".csv",".xlsx",".xls")):
            found_files.append(os.path.join(root,f))
print("\n📌 Detected dataset files:")
for f in found_files:
    print(" -", f)

if len(found_files) == 0:
    raise SystemExit("No CSV/XLSX files found in uploaded ZIP. Please include dataset files and re-run.")

# ---------------------------
# 2) Load files into DataFrames
# ---------------------------
def load_table(path):
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)
    except Exception as e:
        print("Failed to load", path, ":", e)
        return None

dfs = []
for p in found_files:
    df = load_table(p)
    if df is not None:
        print(f"Loaded {p} shape={df.shape}")
        dfs.append(df)

if len(dfs) == 0:
    raise SystemExit("No valid dataset loaded.")

# ---------------------------
# 3) Normalize column names & detect features
# ---------------------------
# Target features requested
FEATURE_ORDER = ["hbg", "mcv", "mch", "mchc", "rbc", "rdw", "hba2", "hbf"]
# create map of possible column name variants -> canonical
possible_map = {
    "hb":"hbg", "hbg":"hbg", "hemoglobin":"hbg",
    "mcv":"mcv",
    "mch":"mch",
    "mchc":"mchc",
    "rbc":"rbc", "rbc count":"rbc",
    "rdw":"rdw", "rdwcv":"rdw", "rdw-cv":"rdw",
    "hba2":"hba2",
    "hb a2":"hba2",
    "hbf":"hbf", "hb f":"hbf",
    "diagnosis":"diagnosis", "label":"diagnosis", "result":"diagnosis"
}

def canonicalize_columns(df):
    cols = {}
    # Create a list of (normalized_possible_map_key, original_possible_map_key, canonical_value)
    # Sort this list by the length of the normalized_possible_map_key in descending order
    # This prioritizes more specific (longer) matches first.
    sorted_possible_matches = []
    for k_orig, v_canonical in possible_map.items():
        normalized_k = k_orig.replace(" ", "").replace("_","").replace("-","")
        sorted_possible_matches.append((normalized_k, k_orig, v_canonical))
    sorted_possible_matches.sort(key=lambda x: len(x[0]), reverse=True)

    for c in df.columns:
        original_col_normalized = c.strip().lower().replace(" ", "").replace("_","").replace("-","")
        mapped_to_canonical = None

        for norm_k_pm, k_orig_pm, v_canonical_pm in sorted_possible_matches:
            if original_col_normalized == norm_k_pm: # Exact match
                mapped_to_canonical = v_canonical_pm
                break
            elif norm_k_pm in original_col_normalized: # e.g., 'mch' in 'mch-value'
                mapped_to_canonical = v_canonical_pm
                break
            elif original_col_normalized in norm_k_pm: # e.g., 'hb' in 'hba2'
                mapped_to_canonical = v_canonical_pm
                break

        if mapped_to_canonical:
            cols[c] = mapped_to_canonical
        else:
            cols[c] = c  # keep as-is

    final_df = df.rename(columns=cols)

    # Ensure column names are unique by appending a suffix if duplicates exist
    seen_cols = {}
    unique_columns = []
    for col in final_df.columns:
        if col in seen_cols:
            suffix = seen_cols[col]
            new_col_name = f"{col}_{suffix}"
            seen_cols[col] += 1
            unique_columns.append(new_col_name)
        else:
            seen_cols[col] = 1
            unique_columns.append(col)
    final_df.columns = unique_columns

    return final_df

# canonicalize and keep loaded
cleaned_dfs = []
for df in dfs:
    df = df.copy()
    df = canonicalize_columns(df)
    cleaned_dfs.append(df)

# Merge into a single DataFrame (outer join on columns)
df_all = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
print("\n✅ Combined raw shape:", df_all.shape)

# ---------------------------
# 4) Ensure feature columns exist, coerce numeric
# ---------------------------
# create 'diagnosis' column if variations exist
if 'diagnosis' not in df_all.columns:
    # try to find any column that looks like diagnosis
    for c in df_all.columns:
        if any(x in c.lower() for x in ["diagnos","label","result","type","phenotype"]):
            df_all.rename(columns={c:'diagnosis'}, inplace=True)
            break

# try cast feature columns to numeric if present
for f in FEATURE_ORDER:
    if f in df_all.columns:
        df_all[f] = pd.to_numeric(df_all[f], errors='coerce')
    else:
        df_all[f] = np.nan  # create column so all features present

print("\n🔍 Columns available now:", [c for c in df_all.columns if c in FEATURE_ORDER or c=='diagnosis'])

# ---------------------------
# 5) Build binary target: presence of thalassemia (YES/NO)
# ---------------------------
def map_label_to_binary(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    # check words that indicate Thalassemia presence
    positive_markers = ['thalassemia','thalassaemia','alpha','beta','carrier','trait','disease','major','minor','intermedia','hp fh','hp fh','hpfh']
    negative_markers = ['normal','healthy','no','none']
    for m in positive_markers:
        if m in s:
            return 1
    for m in negative_markers:
        if m in s:
            return 0
    # fallback: if contains numeric codes? assume 0
    return 0

if 'diagnosis' in df_all.columns:
    df_all['target'] = df_all['diagnosis'].apply(map_label_to_binary)
elif 'diagnosis_1' in df_all.columns: # Added this to handle suffixed diagnosis column if any
    df_all['target'] = df_all['diagnosis_1'].apply(map_label_to_binary)
else:
    # If no diagnosis column, try 'type' or 'phenotype'
    col_try = None
    for c in df_all.columns:
        if any(k in c.lower() for k in ['type','phenotype']):
            col_try = c; break
    if col_try:
        df_all['target'] = df_all[col_try].apply(map_label_to_binary)
    else:
        raise SystemExit("No diagnosis/label column found in datasets. Add a column like 'Diagnosis' with values (Normal/Alpha/Beta/Carrier/etc).")

print("\n📊 Target distribution (before cleaning):")
print(df_all['target'].value_counts())

# ---------------------------
# 6) Keep only the requested features + target and drop rows with all-feature-missing
# ---------------------------
use_cols = FEATURE_ORDER + ['target']
df_features = df_all[use_cols].copy()
# drop rows which miss ALL features
df_features = df_features.dropna(how='all', subset=FEATURE_ORDER)
print("\n✅ Shape after selecting features (rows without all-feature-missing):", df_features.shape)

# Impute numeric columns with mean
for f in FEATURE_ORDER:
    if df_features[f].isnull().any():
        df_features[f].fillna(df_features[f].mean(), inplace=True)

print("\nℹ️ Any remaining nulls:", df_features.isnull().sum().to_dict())

# ---------------------------
# 7) Dataset duplication / augmentation
# ---------------------------
print("\nWould you like to augment (duplicate) the dataset to increase samples? This repeats rows with small noise.")
dup_factor = input("Enter duplication multiplier (1 = no duplication, 2 = double dataset, 3 = triple, ...): ").strip()
try:
    dup_factor = int(dup_factor)
    if dup_factor < 1: dup_factor = 1
except:
    dup_factor = 1

def duplicate_with_noise(df, factor, noise_scale=0.01):
    if factor <= 1:
        return df.copy()
    rows = [df]
    numeric_cols = FEATURE_ORDER
    for i in range(factor-1):
        tmp = df.copy()
        for c in numeric_cols:
            # add small gaussian noise proportional to mean
            scale = max(1e-6, df[c].std()) * noise_scale
            tmp[c] = tmp[c] + np.random.normal(0, scale, size=tmp.shape[0])
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)

df_aug = duplicate_with_noise(df_features, dup_factor)
print("✅ After duplication: shape =", df_aug.shape)
print("Target counts after duplication:")
print(df_aug['target'].value_counts())

# ---------------------------
# 8) Prepare X,y and scaling
# ---------------------------
X = df_aug[FEATURE_ORDER].values
y = df_aug['target'].values

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 9) Handle class imbalance with SMOTE
# ---------------------------
print("\nBalancing classes using SMOTE...")
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_scaled, y)
unique, counts = np.unique(y_bal, return_counts=True)
print("Counts after SMOTE:", dict(zip(unique, counts)))

# ---------------------------
# 10) Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.20, random_state=42, stratify=y_bal)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------
# 11) Define models to train
# ---------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM_rbf": SVC(kernel='rbf', probability=True),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

results = {}
trained_models = {}

print("\n🚀 Training models (this may take a few minutes)...")
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        # roc auc (binary)
        try:
            y_proba = model.predict_proba(X_test)[:,1]
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc = None
        results[name] = {"accuracy":acc, "f1":f1, "roc_auc":roc}
        trained_models[name] = model
        print(f" - {name}: acc={acc:.4f}, f1={f1:.4f}, roc={roc}")
    except Exception as e:
        print(f" - {name} failed: {e}")

# ---------------------------
# 12) Ensemble (soft voting) of top tree models
# ---------------------------
print("\nBuilding soft Voting ensemble of top tree learners...")
ensemble_estimators = []
for nm in ["RandomForest","XGBoost","LightGBM","ExtraTrees"]:
    if nm in trained_models:
        ensemble_estimators.append((nm, trained_models[nm]))
if ensemble_estimators:
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=-1)
    try:
        ensemble.fit(X_train, y_train)
        y_pred_e = ensemble.predict(X_test)
        acc_e = accuracy_score(y_test, y_pred_e)
        f1_e = f1_score(y_test, y_pred_e, average="weighted")
        roc_e = None
        try:
            roc_e = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1])
        except:
            pass
        results["Ensemble"] = {"accuracy":acc_e, "f1":f1_e, "roc_auc":roc_e}
        trained_models["Ensemble"] = ensemble
        print(f" - Ensemble: acc={acc_e:.4f}, f1={f1_e:.4f}, roc={roc_e}")
    except Exception as e:
        print("Ensemble training failed:", e)

# ---------------------------
# 13) Simple CNN (optional, small)
# ---------------------------
print("\nTraining small CNN on tabular features (as 1D)...")
try:
    seq_X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    seq_X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    n_classes = len(np.unique(y_train))
    cnn = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=(seq_X_train.shape[1],1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(seq_X_train, y_train, epochs=12, batch_size=32, verbose=0)
    cnn_acc = cnn.evaluate(seq_X_test, y_test, verbose=0)[1]
    results["CNN"] = {"accuracy": cnn_acc, "f1": None, "roc_auc": None}
    trained_models["CNN"] = cnn
    print(f" - CNN accuracy={cnn_acc:.4f}")
except Exception as e:
    print(" - CNN failed:", e)

# ---------------------------
# 14) Compare results and choose best model by weighted F1
# ---------------------------
res_df = pd.DataFrame(results).T.sort_values(by='f1', ascending=False)
print("\n📈 Model comparison (sorted by weighted F1):\n")
print(res_df)

best_name = res_df.index[0]
best_model = trained_models.get(best_name)
print(f"\n🏆 Best model selected: {best_name}")

# ---------------------------
# 15) Save model bundle (model, scaler, label encoder, feature order)
# ---------------------------
label_encoder = LabelEncoder()
label_encoder.fit(y_bal)  # just numeric 0/1 but keep for consistency

bundle = {
    "model_name": best_name,
    "model": best_model,
    "scaler": scaler,
    "label_encoder": label_encoder,
    "feature_order": FEATURE_ORDER,
    "target_map": {0:"Normal", 1:"Thalassemia"}
}

joblib.dump(bundle, "thalassemia_model_bundle.pkl")
# zip it
import zipfile
with zipfile.ZipFile("thalassemia_model_bundle.zip", "w") as z:
    z.write("thalassemia_model_bundle.pkl")

print("\n✅ Model bundle saved as thalassemia_model_bundle.pkl and zipped to thalassemia_model_bundle.zip")
print("You can download it now (click the link):")
files.download("thalassemia_model_bundle.zip")

# ---------------------------
# 16) Example: How to use saved model (prints usage snippet)
# ---------------------------
print("""
-------------------------
USAGE SNIPPET (example)
-------------------------
import joblib, numpy as np
bundle = joblib.load("thalassemia_model_bundle.pkl")
model = bundle['model']
scaler = bundle['scaler']
features = bundle['feature_order']  # order: {}
# input values must respect that order
sample = np.array([[hb, rbc, mcv, mch, mchc, hba2, hbf, rdw]])  # adjust order per feature_order in bundle
sample_scaled = scaler.transform(sample)
pred = None
try:
    pred = bundle['model'].predict(sample_scaled)[0]
    proba = bundle['model'].predict_proba(sample_scaled)[0] if hasattr(bundle['model'], 'predict_proba') else None
except Exception as e:
    print("Model predict error:", e)
print("Pred:", pred, "->", bundle['target_map'][pred], " probability:", proba)
""".format(FEATURE_ORDER))

print("\nFinished. If you want a minimal Flask / Streamlit front-end that loads the bundle and presents a form, tell me and I'll generate it.")
