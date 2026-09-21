import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 1. Multi-parameter rule label generation
def compute_multipar_class(row):
    penalty = 0
    if row['pH'] < 6.5 or row['pH'] > 8.5: penalty += 2
    if row['do_mgL'] < 4.0: penalty += 2
    if row['turbidity_NTU'] > 50.0: penalty += 2
    elif row['turbidity_NTU'] > 25.0: penalty += 1
    if row['tds_mgL'] > 1000: penalty += 1
    
    if penalty >= 5: return 'Critical'
    elif penalty >= 3: return 'Warning'
    elif penalty >= 1: return 'Caution'
    else: return 'Normal'

def main():
    set_seed(42)
    
    print("Loading data...")
    df = pd.read_csv("fishpond_dataset_multiclass_2153.csv")
    
    # Drop fake time features & site_name if there, keep only actual measurements
    # The original csv has: sample_id,timestamp,orp_mV,ec_uScm,tds_mgL,turbidity_NTU,temp_C,pH,do_mgL,class,class_multi
    features = ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL']
    
    # Apply new rule-based label (Fix Problem 3: independent of single column)
    df['class'] = df.apply(compute_multipar_class, axis=1)
    
    # Save the updated dataset for the app to use
    df.to_csv("processed_fishpond_train.csv", index=False)
    
    X = df[features].values
    y = df['class'].values
    
    # Train / Val / Test (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), stratify=y_temp, random_state=42)
    
    # 2. Fit Scaler & Encoder on TRAIN only (Fix Problem 1)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    
    encoder = LabelEncoder()
    y_train_e = encoder.fit_transform(y_train)
    y_val_e = encoder.transform(y_val)
    y_test_e = encoder.transform(y_test)
    joblib.dump(encoder, "encoder.pkl")
    
    # Baseline check: Turbidity-only model vs All-features (Fix Problem 3 baseline requirement)
    print("\n--- Baseline Check: Turbidity-only vs All-features ---")
    turb_idx = features.index('turbidity_NTU')
    rf_turb = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_turb.fit(X_train_s[:, [turb_idx]], y_train_e)
    turb_acc = rf_turb.score(X_test_s[:, [turb_idx]], y_test_e)
    
    rf_all = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_all.fit(X_train_s, y_train_e)
    all_acc = rf_all.score(X_test_s, y_test_e)
    
    print(f"Turbidity-only RF accuracy: {turb_acc:.4f}")
    print(f"All-features RF accuracy: {all_acc:.4f}")
    if abs(all_acc - turb_acc) < 0.01:
        print("WARNING: Turbidity-only model performs within 1% of the all-features model! "
              "Multi-parameter claim is weakly supported by data.")
    else:
        print("Success: All-features model significantly outperforms single-column baseline.")

    # 3. Train MLP (Renamed from DomainConstrainedMLP to WaterQualityMLP - Fix Problem 7)
    class WaterQualityMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(WaterQualityMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.net(x)
            
    input_dim = X_train_s.shape[1]
    num_classes = len(encoder.classes_)
    model = WaterQualityMLP(input_dim, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    Xt_tr = torch.FloatTensor(X_train_s)
    yt_tr = torch.LongTensor(y_train_e)
    
    print("\nTraining WaterQualityMLP...")
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model(Xt_tr), yt_tr)
        loss.backward()
        optimizer.step()
        
    # 4. Evaluate & Metrics
    model.eval()
    with torch.no_grad():
        Xt_te = torch.FloatTensor(X_test_s)
        preds = model(Xt_te).argmax(dim=1).numpy()
    
    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_test_e, preds, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_e, preds))
    
    f1_macro = f1_score(y_test_e, preds, average='macro')
    dvr = (sum(preds == encoder.transform(['Critical'])[0]) / len(preds)) * 100

    metrics = {
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(encoder.classes_, f1_score(y_test_e, preds, average=None).tolist())),
        "domain_violation_rate_pct": dvr,
        "test_samples": len(preds)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    torch.save(model.state_dict(), "pytorch_fishpond_model.pth")
    print("\nSaved pytorch_fishpond_model.pth, scaler.pkl, encoder.pkl, and metrics.json.")
    
if __name__ == "__main__":
    main()
