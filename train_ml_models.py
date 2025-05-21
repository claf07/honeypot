import os
import joblib
import numpy as np
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Directory to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature columns expected by the classify_request function
FEATURE_COLUMNS = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports"
]

# Categorical columns to encode
CATEGORICAL_COLUMNS = ["proto", "service", "state"]

def load_real_data():
    """Load real attack data from attacks.db"""
    conn = sqlite3.connect('SecurityHoneypot/app/attacks.db')
    query = "SELECT * FROM attacks"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def encode_categorical(data):
    """Encode categorical columns using LabelEncoder"""
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    return data, encoders

def main():
    print("Loading real attack data from attacks.db...")
    data = load_real_data()
    
    # Assume the last column is the target (0 = normal, 1 = attack)
    target = data.iloc[:, -1]
    features = data.iloc[:, :-1]
    
    print("Encoding categorical features...")
    features, encoders = encode_categorical(features)
    
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier with GridSearchCV...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, rf_param_grid, cv=3, scoring='f1')
    grid_rf.fit(X_train, y_train)
    rf = grid_rf.best_estimator_
    rf_pred = rf.predict(X_test)
    print(f"Random Forest Best Parameters: {grid_rf.best_params_}")
    print(f"Random Forest Best F1 Score: {grid_rf.best_score_:.4f}")
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("Training XGBoost Classifier with GridSearchCV...")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid_xgb = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring='f1')
    grid_xgb.fit(X_train, y_train)
    xgb = grid_xgb.best_estimator_
    xgb_pred = xgb.predict(X_test)
    print(f"XGBoost Best Parameters: {grid_xgb.best_params_}")
    print(f"XGBoost Best F1 Score: {grid_xgb.best_score_:.4f}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred))
    
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.3, random_state=42)
    iso_forest.fit(X_train[y_train == 0])
    iso_pred = iso_forest.predict(X_test)
    iso_pred = [1 if x == -1 else 0 for x in iso_pred]
    print("Isolation Forest Classification Report:")
    print(classification_report(y_test, iso_pred))
    
    print("Saving models and encoders...")
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    print("Training complete. Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
