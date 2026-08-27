"""
Micro Climate Crop Advisor — Model Training Script
===================================================
Run this ONCE to generate data and train/save the model:
    python train_model.py

Outputs:
  models/crop_model.pkl       — trained RandomForest classifier
  models/scaler.pkl           — StandardScaler for input normalization
  models/label_encoder.pkl    — LabelEncoder for crop names
  data/crop_data.csv          — synthetic training dataset
"""

import numpy as np
import pandas as pd
import pickle, os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── Seed for reproducibility ───────────────────────────────────────────────
np.random.seed(42)

# ─── Crop profiles (N, P, K, pH, temp°C, humidity%, rainfall mm) ─────────────
CROP_PROFILES = {
    "Rice":        dict(N=(60,100), P=(35,60),  K=(30,50),  pH=(5.5,6.5), T=(20,27), H=(80,95), R=(150,250)),
    "Wheat":       dict(N=(60,100), P=(40,70),  K=(35,55),  pH=(6.0,7.5), T=(12,22), H=(50,70), R=(50,100)),
    "Maize":       dict(N=(70,110), P=(40,65),  K=(30,55),  pH=(5.5,7.0), T=(18,28), H=(55,75), R=(50,120)),
    "Chickpea":    dict(N=(20,45),  P=(55,80),  K=(70,100), pH=(5.5,7.0), T=(16,26), H=(14,40), R=(30,60)),
    "Kidney Beans":dict(N=(15,40),  P=(55,80),  K=(15,30),  pH=(5.5,7.0), T=(18,28), H=(18,58), R=(70,130)),
    "Pigeon Peas": dict(N=(10,30),  P=(55,80),  K=(15,30),  pH=(5.0,7.0), T=(20,30), H=(45,70), R=(60,120)),
    "Moth Beans":  dict(N=(10,30),  P=(40,60),  K=(15,30),  pH=(3.5,6.5), T=(24,36), H=(45,72), R=(30,80)),
    "Mung Bean":   dict(N=(10,30),  P=(35,60),  K=(10,30),  pH=(6.2,7.2), T=(25,35), H=(82,92), R=(45,85)),
    "Black Gram":  dict(N=(25,45),  P=(55,75),  K=(15,35),  pH=(5.5,7.5), T=(25,35), H=(65,80), R=(45,80)),
    "Lentil":      dict(N=(10,30),  P=(55,75),  K=(15,30),  pH=(6.0,7.5), T=(15,25), H=(64,74), R=(35,65)),
    "Pomegranate": dict(N=(18,35),  P=(15,30),  K=(35,55),  pH=(5.5,7.0), T=(22,30), H=(90,95), R=(100,200)),
    "Banana":      dict(N=(80,120), P=(70,90),  K=(45,65),  pH=(5.0,6.5), T=(25,30), H=(75,90), R=(100,200)),
    "Mango":       dict(N=(15,35),  P=(15,30),  K=(25,45),  pH=(5.5,7.5), T=(24,32), H=(45,70), R=(90,180)),
    "Grapes":      dict(N=(18,35),  P=(15,30),  K=(40,60),  pH=(5.5,7.0), T=(22,32), H=(80,90), R=(100,200)),
    "Watermelon":  dict(N=(80,110), P=(50,70),  K=(40,65),  pH=(5.5,7.0), T=(24,32), H=(80,90), R=(50,100)),
    "Muskmelon":   dict(N=(80,110), P=(50,70),  K=(40,65),  pH=(6.0,7.0), T=(26,36), H=(92,95), R=(20,50)),
    "Apple":       dict(N=(30,55),  P=(125,145),K=(195,220),pH=(5.5,7.0), T=(6,18),  H=(90,95), R=(100,200)),
    "Orange":      dict(N=(15,35),  P=(15,30),  K=(10,25),  pH=(6.0,7.5), T=(22,30), H=(80,90), R=(100,200)),
    "Papaya":      dict(N=(40,60),  P=(55,75),  K=(40,60),  pH=(6.5,7.5), T=(25,35), H=(90,95), R=(120,220)),
    "Coconut":     dict(N=(20,40),  P=(20,40),  K=(25,45),  pH=(5.0,8.0), T=(20,32), H=(88,95), R=(100,200)),
    "Cotton":      dict(N=(90,140), P=(40,60),  K=(15,30),  pH=(6.0,8.0), T=(20,30), H=(75,85), R=(60,120)),
    "Jute":        dict(N=(55,85),  P=(20,40),  K=(15,30),  pH=(6.0,7.0), T=(24,37), H=(70,90), R=(160,250)),
    "Coffee":      dict(N=(80,110), P=(35,55),  K=(25,45),  pH=(6.0,7.0), T=(15,25), H=(55,75), R=(150,250)),
}

# ─── Generate synthetic samples ──────────────────────────────────────────────
SAMPLES_PER_CROP = 100
records = []
for crop, p in CROP_PROFILES.items():
    for _ in range(SAMPLES_PER_CROP):
        row = {
            "N":        np.random.uniform(*p["N"]),
            "P":        np.random.uniform(*p["P"]),
            "K":        np.random.uniform(*p["K"]),
            "ph":       np.random.uniform(*p["pH"]),
            "temperature": np.random.uniform(*p["T"]),
            "humidity":    np.random.uniform(*p["H"]),
            "rainfall":    np.random.uniform(*p["R"]),
            "label":    crop,
        }
        records.append(row)

df = pd.DataFrame(records)
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
df.to_csv("data/crop_data.csv", index=False)
print(f"Dataset saved → {len(df)} rows, {df['label'].nunique()} crops")

# ─── Prepare features / labels ───────────────────────────────────────────────
FEATURES = ["N","P","K","ph","temperature","humidity","rainfall"]
X = df[FEATURES].values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# ─── Train Random Forest ──────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv  = cross_val_score(rf, X_sc, y_enc, cv=5, scoring="accuracy")

print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"CV Accuracy   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─── Save artefacts ───────────────────────────────────────────────────────────
with open("models/crop_model.pkl",    "wb") as f: pickle.dump(rf, f)
with open("models/scaler.pkl",        "wb") as f: pickle.dump(scaler, f)
with open("models/label_encoder.pkl", "wb") as f: pickle.dump(le, f)

# Save class names for frontend
with open("models/classes.json", "w") as f:
    json.dump(le.classes_.tolist(), f)

# Feature importances
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
for feat, imp in fi.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<15} {bar} {imp:.4f}")

print("\n✅ All model artefacts saved to models/")
