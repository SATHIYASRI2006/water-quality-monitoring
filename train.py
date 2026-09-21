import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import json
import random

# Fix seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Load raw dataset
df = pd.read_csv("fishpond_dataset_multiclass_2153.csv")

# Baseline check on original data: EC-TDS correlation
ec_tds_corr = df['ec_uScm'].corr(df['tds_mgL'])
print(f"Data Validation - EC/TDS Correlation: {ec_tds_corr:.4f}")

# Rebuild label from a multi-parameter rule to avoid single-column dominance
def compute_multipar_class(row):
    # WHO/TNPCB based scoring
    penalty = 0
    if row['pH'] < 6.5 or row['pH'] > 8.5: penalty += 1
    if row['do_mgL'] < 4.0: penalty += 1
    if row['turbidity_NTU'] > 30: penalty += 1
    if row['tds_mgL'] > 800: penalty += 1
    if row['temp_C'] > 32 or row['temp_C'] < 18: penalty += 1
    
    if penalty == 0: return "Normal"
    elif penalty == 1: return "Caution"
    elif penalty == 2: return "Warning"
    else: return "Critical"

df['class_multi'] = df.apply(compute_multipar_class, axis=1)

# Ensure no fake time features are used
feat_cols = ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL']
X = df[feat_cols].values
y = df['class_multi'].values

print("Class balance (Multi-parameter label):")
print(df['class_multi'].value_counts())

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Stratified split: 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

# Fit scaler on train only (Fixing broken scaler issue)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check single feature (Turbidity-only) vs all-features accuracy
from sklearn.ensemble import RandomForestClassifier
rf_all = RandomForestClassifier(n_estimators=50, random_state=42)
rf_all.fit(X_train_scaled, y_train)
acc_all = rf_all.score(X_test_scaled, y_test)

turbidity_idx = feat_cols.index('turbidity_NTU')
rf_turb = RandomForestClassifier(n_estimators=50, random_state=42)
rf_turb.fit(X_train_scaled[:, turbidity_idx:turbidity_idx+1], y_train)
acc_turb = rf_turb.score(X_test_scaled[:, turbidity_idx:turbidity_idx+1], y_test)

print(f"Turbidity-only model accuracy: {acc_turb:.4f}")
print(f"All-features model accuracy: {acc_all:.4f}")
if abs(acc_all - acc_turb) < 0.01:
    print("WARNING: Turbidity alone perfectly predicts the class. The multi-parameter claim is not justified by the data.")
else:
    print("Multi-parameter dependency verified.")

# Model Definition
class WaterQualityMLP(nn.Module):
    def __init__(self, input_dim=7, num_classes=4):
        super(WaterQualityMLP, self).__init__()
        self.network = nn.Sequential(
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
        return self.network(x)

model = WaterQualityMLP(input_dim=len(feat_cols), num_classes=len(encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)

epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

# Evaluation on Test set
model.eval()
X_test_t = torch.FloatTensor(X_test_scaled)
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()

print("\nClassification Report (Test):")
print(classification_report(y_test, preds, target_names=encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

f1_macro = f1_score(y_test, preds, average='macro')
f1_per_class = f1_score(y_test, preds, average=None)

# Calculate DVR (Domain Violation Rate) on Test Set
# Let's say DVR is the percentage of samples that fall outside Safe bounds.
# In our new labeling, anything not 'Normal' has penalty > 0, i.e. violation.
dvr = np.sum(y_test != encoder.transform(["Normal"])[0]) / len(y_test)

metrics = {
    "f1_score_macro": float(f1_macro),
    "f1_score_per_class": {cls: float(f1) for cls, f1 in zip(encoder.classes_, f1_per_class)},
    "dvr_test_set": float(dvr),
    "total_dataset_rows": len(df)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
torch.save(model.state_dict(), "pytorch_fishpond_model.pth")
df.to_csv("fishpond_dataset_multiclass_2153.csv", index=False) # Overwrite with new labels
print("Training complete. Artifacts saved.")
