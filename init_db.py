import sqlite3
import os

DB_PATH = "water_quality.db"

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE sensor_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            site_name TEXT,
            orp_mV REAL,
            ec_uScm REAL,
            tds_mgL REAL,
            turbidity_NTU REAL,
            temp_C REAL,
            pH REAL,
            do_mgL REAL,
            prediction_class TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()
