import pandas as pd
import numpy as np

df = pd.read_csv("fishpond_dataset_multiclass_2153.csv")

print("--- Data Validation Report ---")

# EC-TDS Correlation
corr = df["ec_uScm"].corr(df["tds_mgL"])
print(f"EC-TDS Correlation: {corr:.4f}")

# Class Balance
print("\nClass Balance (Original 'class' column):")
print(df["class"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

print("\nClass Balance (Original 'class_multi' column):")
if "class_multi" in df.columns:
    print(df["class_multi"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
else:
    print("Not found")

# Value Ranges
print("\nValue Ranges:")
features = ["orp_mV", "ec_uScm", "tds_mgL", "turbidity_NTU", "temp_C", "pH", "do_mgL"]
for f in features:
    print(f"  {f}: Min={df[f].min():.2f}, Max={df[f].max():.2f}, Mean={df[f].mean():.2f}")
