import sqlite3
import time
import random
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from model_backend import load_pytorch_artifacts, predict_with_pytorch

warnings.filterwarnings("ignore")

DB_PATH = "water_quality.db"
SITES = ["Tank A (Main Reservoir)", "Industrial Basin B", "Effluent Plant C", "Coastal Bio-Swale", "Metro Reservoir A"]

def run_simulator():
    print("Starting IoT Simulator...")
    model, scaler, encoder, df, feat_cols = load_pytorch_artifacts()
    
    # We will sample from real dataset to ensure realistic values, but add slight noise
    base_data = df.copy()
    
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    
    while True:
        for site in SITES:
            # Pick a random row
            row = base_data.sample(1).iloc[0]
            
            raw_feats = [float(row[col]) for col in feat_cols]
            
            # Predict
            pred_res = predict_with_pytorch(raw_feats)
            pred_class = pred_res["status_tier"]
            conf = pred_res["confidence"]
            
            # Insert into DB
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute('''
                INSERT INTO sensor_telemetry (timestamp, site_name, orp_mV, ec_uScm, tds_mgL, turbidity_NTU, temp_C, pH, do_mgL, prediction_class, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (now_str, site, raw_feats[0], raw_feats[1], raw_feats[2], raw_feats[3], raw_feats[4], raw_feats[5], raw_feats[6], pred_class, conf))
            
        conn.commit()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Inserted 5 records (1 per site)...")
        time.sleep(3) # simulate every 3 seconds

if __name__ == "__main__":
    run_simulator()
