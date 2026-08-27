# 🌾 Micro Climate Crop Advisor
### AI-Powered Smart Farming System

A full-stack Flask + ML application that recommends optimal crops based on soil nutrients, real-time weather data, and historical patterns — with smart planning and risk mitigation.

---

## 🗂 Project Structure

```
crop_advisor/
├── app.py                  # Flask backend (all routes + API)
├── train_model.py          # ML training script (run once)
├── requirements.txt
│
├── models/                 # Auto-created by train_model.py
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── classes.json
│
├── data/
│   └── crop_data.csv       # Generated training dataset
│
├── templates/
│   ├── base.html           # Shared layout + navbar
│   ├── index.html          # Landing page
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html      # Weather + recent predictions
│   ├── advisor.html        # Main crop advisor form
│   └── history.html        # Past predictions table
│
└── static/
    ├── css/style.css       # Full design system
    └── js/main.js
```

---

## ⚙️ Setup — Step by Step

### Step 1: Clone / create project folder
```bash
cd ~/projects
# (files are already here if you downloaded the zip)
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get a free OpenWeatherMap API key
1. Go to https://openweathermap.org/api → Sign up free
2. Navigate to **API Keys** in your account
3. Copy your API key

### Step 5: Set environment variables
```bash
# Linux / Mac
export WEATHER_API_KEY="your_key_here"
export SECRET_KEY="any-random-secret"

# Windows (PowerShell)
$env:WEATHER_API_KEY = "your_key_here"
$env:SECRET_KEY      = "any-random-secret"
```

Or create a `.env` file:
```
WEATHER_API_KEY=your_key_here
SECRET_KEY=my-super-secret-key
```
Then install `python-dotenv` (already in requirements) and add to app.py top:
```python
from dotenv import load_dotenv; load_dotenv()
```

### Step 6: Train the ML model (run once)
```bash
python train_model.py
```
Expected output:
```
Dataset saved → 2200 rows, 22 crops
Test Accuracy : 97.27%
CV Accuracy   : 96.86% ± 0.48%
✅ All model artefacts saved to models/
```

### Step 7: Launch the Flask app
```bash
python app.py
```
Visit → http://localhost:5000

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Run crop ML prediction |
| POST | `/api/weather` | Fetch live weather for a location |
| POST | `/api/risk` | Assess risks for given conditions |
| POST | `/api/plan` | Get smart crop plan |
| GET  | `/api/history` | Fetch user prediction history |
| POST | `/api/profile` | Update user location |

### Example: `/api/predict`
```json
POST /api/predict
{
  "N": 80, "P": 45, "K": 40, "ph": 6.5,
  "temperature": 25, "humidity": 70, "rainfall": 100
}

Response:
{
  "predictions": [
    {"crop": "Rice", "probability": 85.2, "rank": 1},
    {"crop": "Maize", "probability": 8.4, "rank": 2},
    {"crop": "Wheat", "probability": 4.1, "rank": 3}
  ],
  "plan": { "sow_period": "June-July", "harvest_period": "Oct-Nov", ... },
  "risks": [{ "type": "Conditions Optimal", "severity": "Low", ... }]
}
```

---

## 🧠 ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Training samples | 2,200 (100 per crop × 22 crops) |
| Features | N, P, K, pH, Temperature, Humidity, Rainfall |
| Output | Top-3 crops with probability scores |
| Evaluation | Accuracy ~97%, 5-fold CV ~97% |

**Feature Importances** (typical):
1. Temperature (~22%)
2. Rainfall (~20%)
3. Humidity (~18%)
4. K — Potassium (~14%)
5. N — Nitrogen (~12%)
6. P — Phosphorus (~9%)
7. pH (~5%)

---

## 🌾 Supported Crops (22)

Rice, Wheat, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans,
Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes,
Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 🗄 Database Schema

```sql
Users       (id, name, email, password, location, created_at)
SoilData    (id, user_id, N, P, K, pH, saved_at)
WeatherData (id, user_id, location, temperature, humidity, rainfall, description, fetched_at)
Predictions (id, user_id, crop, probability, inputs_json, predicted_at)
```

---

## 🚀 Deployment (Production)

### Using Gunicorn + Nginx (Linux VPS)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Railway / Render (Free cloud)
1. Push to GitHub
2. Connect to Railway / Render
3. Set environment variables in dashboard
4. Deploy — done!

### Switch to MySQL for production
```python
# In app.py change:
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://user:pass@localhost/crop_advisor"
```

---

## 📦 Real Kaggle Dataset (Optional Upgrade)

To use the real Kaggle crop recommendation dataset:
1. Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Place `Crop_recommendation.csv` in `data/`
3. Modify `train_model.py` to load it:
```python
df = pd.read_csv("data/Crop_recommendation.csv")
```
The column names are identical (N, P, K, ph, temperature, humidity, rainfall, label).

---

## ⚠️ Risk Detection Logic

| Condition | Risk Type | Severity |
|-----------|-----------|----------|
| Rainfall < 40mm | Drought | High |
| Rainfall > 200mm | Flood | High |
| Temperature > 38°C | Heat Stress | Medium |
| Temperature < 10°C | Cold Stress | Medium |
| Humidity > 90% | Fungal Disease | Medium |
| All within range | Optimal | Low |

---

## 📝 Project Report Outline

1. Introduction & Problem Statement
2. System Architecture
3. Dataset Description
4. ML Model: Random Forest Classifier
5. Feature Engineering & Preprocessing
6. Model Evaluation (Accuracy, Confusion Matrix)
7. Flask Backend Design
8. Frontend UI Design
9. Weather API Integration
10. Smart Planning System
11. Risk Mitigation Module
12. Database Design
13. Testing & Results
14. Conclusion & Future Work

---

*Built with Flask, Scikit-learn, SQLAlchemy, Chart.js, and OpenWeatherMap API*
