"""
Micro Climate Crop Advisor — Flask Backend (corrected)
=======================================================
Fixes applied vs the original:
  1. Login route: session.clear() before setting new keys so stale data from
     a previous user / session never leaks into the new session.
  2. api_predict: SoilData always uses db.session.add() (INSERT), never merge()
     or update-in-place, so every prediction produces a new audit row.
  3. api_predict: Prediction always uses db.session.add() (INSERT).
     Each call to /api/predict creates exactly one new Prediction row and one
     new SoilData row regardless of how many times the same user submits.
  4. api_weather: WeatherRecord always uses db.session.add() (INSERT) — one
     new row per fetch, building a full weather history log.
  5. api_weather: the location fallback no longer injects "Hyderabad, India"
     — if no location is provided an error is returned so the frontend can
     prompt the user explicitly.
  6. api_profile: still UPDATEs the User.location column (correct — only one
     "current location" per user is stored there), but now also flushes the
     session key so any cached value is immediately replaced.
"""

import os, json, pickle, hashlib, requests
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
from statistics import mean
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, flash)
from flask_sqlalchemy import SQLAlchemy

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "crop-advisor-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///crop_advisor.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Custom Jinja2 filter: parse stored JSON strings in templates
import json as _json
@app.template_filter("fromjson")
def fromjson_filter(s):
    try:
        return _json.loads(s)
    except Exception:
        return {}

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "ae6b7b9b203ea0c709fc4f5b9267d039")
WEATHER_BASE    = "https://api.openweathermap.org/data/2.5"

# ── Load ML artefacts ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

model  = _load("crop_model.pkl")
scaler = _load("scaler.pkl")
le     = _load("label_encoder.pkl")

classes = []
classes_path = os.path.join(MODEL_DIR, "classes.json")
if os.path.exists(classes_path):
    with open(classes_path) as f:
        classes = json.load(f)

# ADD after classes loading
_meta_path = os.path.join(MODEL_DIR, "model_meta.json")
model_meta = json.load(open(_meta_path)) if os.path.exists(_meta_path) else {}
# ── Database models ───────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = "users"
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(64),  nullable=False)   # sha-256 hex
    location   = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    soil_data   = db.relationship("SoilData",      backref="user", lazy=True)
    predictions = db.relationship("Prediction",    backref="user", lazy=True)
    weather_log = db.relationship("WeatherRecord", backref="user", lazy=True)


class SoilData(db.Model):
    """
    One row per prediction submission — full audit trail.
    FIX: never merge/update an existing row; always INSERT a new one.
    """
    __tablename__ = "soil_data"
    id      = db.Column(db.Integer, primary_key=True)   # auto-increment PK
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    N       = db.Column(db.Float)
    P       = db.Column(db.Float)
    K       = db.Column(db.Float)
    pH      = db.Column(db.Float)
    saved_at= db.Column(db.DateTime, default=datetime.utcnow)


class WeatherRecord(db.Model):
    """
    One row per weather fetch — full fetch history.
    FIX: always INSERT; never overwrite the previous record.
    """
    __tablename__ = "weather_data"
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"))
    location    = db.Column(db.String(200))
    temperature = db.Column(db.Float)
    humidity    = db.Column(db.Float)
    rainfall    = db.Column(db.Float)
    description = db.Column(db.String(200))
    fetched_at  = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    """
    One row per /api/predict call — full prediction history.
    FIX: always INSERT; never overwrite the previous prediction.
    """
    __tablename__ = "predictions"
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    crop        = db.Column(db.String(100))
    probability = db.Column(db.Float)
    inputs_json = db.Column(db.Text)   # full 7-field payload as JSON
    predicted_at= db.Column(db.DateTime, default=datetime.utcnow)
# ── NEW: Feedback model ───────────────────────────────────────────────────────
class Feedback(db.Model):
    __tablename__ = "feedback"
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    prediction_id= db.Column(db.Integer, db.ForeignKey("predictions.id"))
    crop_planted = db.Column(db.String(100))   # what farmer actually planted
    rating       = db.Column(db.Integer)        # 1-5 stars
    comment      = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

# ── NEW: Farmer mode tip model (pre-loaded from JSON) ────────────────────────
# Tips are read-only; stored in farmer_tips.json, not the DB

# ── Auth helpers ──────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Weather helpers ───────────────────────────────────────────────────────────
def fetch_weather(location: str) -> dict:
    """Return current weather dict, or a dict with an 'error' key."""
    try:
        r = requests.get(
            f"{WEATHER_BASE}/weather",
            params={"q": location, "appid": WEATHER_API_KEY,
                    "units": "metric"},
            timeout=8
        )
        if r.status_code != 200:
            return {"error": f"Weather API returned status {r.status_code}"}
        d = r.json()
        rain = (d.get("rain") or {}).get("1h") or (d.get("rain") or {}).get("3h") or 0.0
        return {
            "temperature": round(d["main"]["temp"],       1),
            "humidity":    round(d["main"]["humidity"],   1),
            "rainfall":    round(float(rain),             2),
            "description": d["weather"][0]["description"].title(),
            "city":        d["name"],
            "country":     d["sys"]["country"],
            "icon":        d["weather"][0]["icon"],
            "wind_speed":  round(d["wind"]["speed"],      1),
            "pressure":    d["main"]["pressure"],
            "feels_like":  round(d["main"]["feels_like"], 1),
        }
    except Exception as ex:
        return {"error": str(ex)}


def fetch_forecast(location: str) -> list:
    """Return up to 40 three-hourly forecast entries."""
    try:
        r = requests.get(
            f"{WEATHER_BASE}/forecast",
            params={"q": location, "appid": WEATHER_API_KEY,
                    "units": "metric", "cnt": 40},
            timeout=8
        )
        if r.status_code != 200:
            return []
        return [
            {
                "datetime":    item["dt_txt"],
                "temperature": round(item["main"]["temp"],     1),
                "humidity":    round(item["main"]["humidity"], 1),
                "rainfall":    round((item.get("rain") or {}).get("3h") or 0.0, 2),
                "description": item["weather"][0]["description"].title(),
            }
            for item in r.json().get("list", [])
        ]
    except Exception:
        return []


# ── ML prediction ─────────────────────────────────────────────────────────────
def predict_crops(N, P, K, pH, temperature, humidity, rainfall) -> list:
    """Return top-3 crops with confidence scores."""
    if model is None:
        return [{"crop": "Model not loaded — run train_model.py first",
                 "probability": 0.0, "rank": 1}]
    feat    = np.array([[float(N), float(P), float(K), float(pH),
                         float(temperature), float(humidity), float(rainfall)]])
    feat_sc = scaler.transform(feat)
    probs   = model.predict_proba(feat_sc)[0]
    top3    = np.argsort(probs)[::-1][:3]
    return [
        {"crop": le.classes_[i],
         "probability": round(float(probs[i]) * 100, 2),
         "rank": r + 1}
        for r, i in enumerate(top3)
    ]
# ── NEW: Forecast-based prediction ───────────────────────────────────────────
def predict_from_forecast(N, P, K, pH, forecast: list) -> list:
    """
    Average temperature, humidity, rainfall across the next 5 forecast days,
    then run the standard ML prediction with those averaged weather values.
    Returns same format as predict_crops().
    """
    if not forecast:
        return []
    temps = [f["temperature"] for f in forecast[:40]]
    hums  = [f["humidity"]    for f in forecast[:40]]
    rains = [f["rainfall"]    for f in forecast[:40]]
    return predict_crops(N, P, K, pH, mean(temps), mean(hums), sum(rains))

# ── Explainable AI ────────────────────────────────────────────────────────────
# Crop ideal-range reference used only for rule-based explanations.
_CROP_RANGES = {
    "Rice":         dict(T=(20,27), H=(80,95), R=(150,250), pH=(5.5,6.5), N=(60,100), P=(35,60),  K=(30,50)),
    "Wheat":        dict(T=(12,22), H=(50,70), R=(50,100),  pH=(6.0,7.5), N=(60,100), P=(40,70),  K=(35,55)),
    "Maize":        dict(T=(18,28), H=(55,75), R=(50,120),  pH=(5.5,7.0), N=(70,110), P=(40,65),  K=(30,55)),
    "Chickpea":     dict(T=(16,26), H=(14,40), R=(30,60),   pH=(5.5,7.0), N=(20,45),  P=(55,80),  K=(70,100)),
    "Cotton":       dict(T=(20,30), H=(75,85), R=(60,120),  pH=(6.0,8.0), N=(90,140), P=(40,60),  K=(15,30)),
    "Mango":        dict(T=(24,32), H=(45,70), R=(90,180),  pH=(5.5,7.5), N=(15,35),  P=(15,30),  K=(25,45)),
    "Banana":       dict(T=(25,30), H=(75,90), R=(100,200), pH=(5.0,6.5), N=(80,120), P=(70,90),  K=(45,65)),
    "Coffee":       dict(T=(15,25), H=(55,75), R=(150,250), pH=(6.0,7.0), N=(80,110), P=(35,55),  K=(25,45)),
    "Coconut":      dict(T=(20,32), H=(88,95), R=(100,200), pH=(5.0,8.0), N=(20,40),  P=(20,40),  K=(25,45)),
    "Jute":         dict(T=(24,37), H=(70,90), R=(160,250), pH=(6.0,7.0), N=(55,85),  P=(20,40),  K=(15,30)),
    "Lentil":       dict(T=(15,25), H=(64,74), R=(35,65),   pH=(6.0,7.5), N=(10,30),  P=(55,75),  K=(15,30)),
    "Pomegranate":  dict(T=(22,30), H=(90,95), R=(100,200), pH=(5.5,7.0), N=(18,35),  P=(15,30),  K=(35,55)),
    "Grapes":       dict(T=(22,32), H=(80,90), R=(100,200), pH=(5.5,7.0), N=(18,35),  P=(15,30),  K=(40,60)),
    "Watermelon":   dict(T=(24,32), H=(80,90), R=(50,100),  pH=(5.5,7.0), N=(80,110), P=(50,70),  K=(40,65)),
    "Apple":        dict(T=(6,18),  H=(90,95), R=(100,200), pH=(5.5,7.0), N=(30,55),  P=(125,145),K=(195,220)),
    "Orange":       dict(T=(22,30), H=(80,90), R=(100,200), pH=(6.0,7.5), N=(15,35),  P=(15,30),  K=(10,25)),
    "Papaya":       dict(T=(25,35), H=(90,95), R=(120,220), pH=(6.5,7.5), N=(40,60),  P=(55,75),  K=(40,60)),
    "Muskmelon":    dict(T=(26,36), H=(92,95), R=(20,50),   pH=(6.0,7.0), N=(80,110), P=(50,70),  K=(40,65)),
    "Black Gram":   dict(T=(25,35), H=(65,80), R=(45,80),   pH=(5.5,7.5), N=(25,45),  P=(55,75),  K=(15,35)),
    "Mung Bean":    dict(T=(25,35), H=(82,92), R=(45,85),   pH=(6.2,7.2), N=(10,30),  P=(35,60),  K=(10,30)),
    "Kidney Beans": dict(T=(18,28), H=(18,58), R=(70,130),  pH=(5.5,7.0), N=(15,40),  P=(55,80),  K=(15,30)),
    "Pigeon Peas":  dict(T=(20,30), H=(45,70), R=(60,120),  pH=(5.0,7.0), N=(10,30),  P=(55,80),  K=(15,30)),
    "Moth Beans":   dict(T=(24,36), H=(45,72), R=(30,80),   pH=(3.5,6.5), N=(10,30),  P=(40,60),  K=(15,30)),
}
def _in_range(value, lo, hi):
    return lo <= value <= hi

def explain_recommendation(crop: str, N, P, K, pH, temperature, humidity, rainfall) -> dict:
    """
    Generate rule-based, human-readable reasons why `crop` was recommended.
    Returns a dict with:
      - reasons: list of positive match strings
      - warnings: list of borderline / mismatch strings
      - confidence_label: 'Highly Recommended' | 'Suitable' | 'Risky'
    """
    r = _CROP_RANGES.get(crop)
    reasons, warnings = [], []

    if r:
        checks = [
            ("temperature", temperature, r["T"], "°C",    "Temperature"),
            ("humidity",    humidity,    r["H"], "%",     "Humidity"),
            ("rainfall",    rainfall,    r["R"], " mm",   "Rainfall"),
            ("ph",          pH,          r["pH"],"",      "Soil pH"),
            ("N",           N,           r["N"], " kg/ha","Nitrogen (N)"),
            ("P",           P,           r["P"], " kg/ha","Phosphorus (P)"),
            ("K",           K,           r["K"], " kg/ha","Potassium (K)"),
        ]
        for key, val, (lo, hi), unit, label in checks:
            if _in_range(val, lo, hi):
                reasons.append(
                    f"✅ {label} ({val}{unit}) is within the ideal range "
                    f"({lo}–{hi}{unit}) for {crop}."
                )
            else:
                direction = "low" if val < lo else "high"
                warnings.append(
                    f"⚠️ {label} ({val}{unit}) is {direction} — "
                    f"ideal range is {lo}–{hi}{unit}."
                )
    else:
        reasons.append(f"✅ {crop} was selected based on the ML model's analysis of all 7 parameters.")

    return {"reasons": reasons, "warnings": warnings}


def confidence_label(probability: float) -> str:
    """Convert a 0–100 probability to a human-readable confidence tier."""
    if probability >= 60:
        return "Highly Recommended"
    elif probability >= 30:
        return "Suitable"
    else:
        return "Risky"
# ── Smart planning ────────────────────────────────────────────────────────────
CROP_CALENDAR = {
    "Rice":        {"sow": "June-July",    "harvest": "Oct-Nov",      "irrigation": "Flood / every 3-4 days",  "duration_days": 120},
    "Wheat":       {"sow": "Oct-Nov",      "harvest": "Mar-Apr",      "irrigation": "5-6 irrigations total",   "duration_days": 150},
    "Maize":       {"sow": "June-July",    "harvest": "Sep-Oct",      "irrigation": "Every 10 days",           "duration_days": 90},
    "Chickpea":    {"sow": "Oct-Nov",      "harvest": "Feb-Mar",      "irrigation": "Minimal (2-3 only)",      "duration_days": 100},
    "Cotton":      {"sow": "Apr-May",      "harvest": "Nov-Jan",      "irrigation": "Every 10-14 days",        "duration_days": 200},
    "Mango":       {"sow": "Jun-Aug",      "harvest": "Apr-Jun",      "irrigation": "Drip / weekly",           "duration_days": 365},
    "Banana":      {"sow": "Jun-Jul",      "harvest": "10-14 months", "irrigation": "Daily drip",              "duration_days": 365},
    "Coffee":      {"sow": "Jun-Jul",      "harvest": "Nov-Feb",      "irrigation": "Weekly drip",             "duration_days": 365},
    "default":     {"sow": "Monsoon onset","harvest": "Post-monsoon", "irrigation": "As needed",               "duration_days": 120},
}

def get_smart_plan(crop: str, location: str, weather: dict) -> dict:
    cal          = CROP_CALENDAR.get(crop, CROP_CALENDAR["default"])
    harvest_date = datetime.utcnow() + timedelta(days=cal["duration_days"])
    return {
        "crop":              crop,
        "sow_period":        cal["sow"],
        "harvest_period":    cal["harvest"],
        "harvest_estimated": harvest_date.strftime("%B %Y"),
        "irrigation":        cal["irrigation"],
        "duration_days":     cal["duration_days"],
        "tips": [
            f"Prepare soil at least 2 weeks before sowing {crop}.",
            "Use certified seeds for better yield and disease resistance.",
            "Monitor weather forecasts weekly during the growth period.",
            f"Follow the irrigation schedule: {cal['irrigation']}.",
        ],
    }


# ── Risk detection ────────────────────────────────────────────────────────────
def assess_risks(temperature: float, humidity: float,
                 rainfall: float, crop: str) -> list:
    risks = []
    if rainfall < 40:
        risks.append({
            "type": "Drought Risk", "severity": "High", "icon": "🌵",
            "description": f"Rainfall ({rainfall} mm) is critically low.",
            "actions": [
                "Install drip irrigation immediately",
                "Mulch soil surface to reduce moisture loss",
                "Consider drought-tolerant varieties",
            ],
        })
    elif rainfall > 200:
        risks.append({
            "type": "Flood Risk", "severity": "High", "icon": "🌊",
            "description": f"Rainfall ({rainfall} mm) is excessively high.",
            "actions": [
                "Ensure drainage channels are clear",
                "Delay sowing by 1-2 weeks",
                "Use raised bed cultivation",
            ],
        })

    if temperature > 38:
        risks.append({
            "type": "Heat Stress", "severity": "Medium", "icon": "🌡️",
            "description": f"Temperature ({temperature}°C) may cause heat stress.",
            "actions": [
                "Irrigate in early morning or evening only",
                "Apply mulch to cool the root zone",
                "Use shade nets for sensitive crops",
            ],
        })
    elif temperature < 10:
        risks.append({
            "type": "Cold Stress", "severity": "Medium", "icon": "❄️",
            "description": f"Temperature ({temperature}°C) is below the optimal range.",
            "actions": [
                "Delay sowing until temperatures rise above 12°C",
                "Use cold-tolerant seed varieties",
                "Apply row covers overnight",
            ],
        })

    if humidity > 90:
        risks.append({
            "type": "Fungal Disease Risk", "severity": "Medium", "icon": "🍄",
            "description": f"High humidity ({humidity}%) promotes fungal diseases.",
            "actions": [
                "Apply preventive fungicide spray",
                "Increase plant spacing for better airflow",
                "Avoid overhead irrigation",
            ],
        })

    if not risks:
        risks.append({
            "type": "Conditions Optimal", "severity": "Low", "icon": "✅",
            "description": "Current weather conditions are within the safe range.",
            "actions": [
                "Proceed with the normal farming schedule",
                "Monitor conditions weekly for any changes",
            ],
        })
    return risks


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name",     "").strip()
        email    = request.form.get("email",    "").strip().lower()
        password = request.form.get("password", "")
        location = request.form.get("location", "").strip()
        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_template("register.html")
        user = User(name=name, email=email,
                    password=hash_password(password), location=location)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email",    "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(
            email=email, password=hash_password(password)
        ).first()
        if not user:
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        # FIX: clear the entire session before writing new keys so that
        # stale data from a previous login (different user, old location, etc.)
        # is never carried forward.
        session.clear()
        session["user_id"]   = user.id
        session["user_name"] = user.name
        # FIX: store only what exists in the DB; never fall back to a
        # hardcoded city string — that caused the "Hyderabad" pre-fill bug.
        session["location"]  = user.location or ""
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    user = db.session.get(User, session["user_id"])

    # 👇 ADD THIS HERE
    if not user:
        session.clear()
        return redirect(url_for("login"))

    recent_preds = (
        Prediction.query
        .filter_by(user_id=user.id)
        .order_by(Prediction.predicted_at.desc())
        .limit(5).all()
    )

    return render_template("dashboard.html", user=user, predictions=recent_preds)



@app.route("/advisor")
@login_required
def advisor():
    return render_template("advisor.html")


@app.route("/history")
@login_required
def history():
    preds = (
        Prediction.query
        .filter_by(user_id=session["user_id"])
        .order_by(Prediction.predicted_at.desc())
        .all()
    )
    return render_template("history.html", predictions=preds)


# ── JSON API endpoints ────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json() or request.form
    try:
        N        = float(data["N"])
        P        = float(data["P"])
        K        = float(data["K"])
        pH       = float(data["ph"])
        temp     = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    # Run ML model
    top3 = predict_crops(N, P, K, pH, temp, humidity, rainfall)

    # FIX: always INSERT a new Prediction row — never update an existing one.
    # db.session.add() maps to SQL INSERT, giving every prediction its own row.
    pred_row = Prediction(
        user_id     = session["user_id"],
        crop        = top3[0]["crop"],
        probability = top3[0]["probability"],
        inputs_json = json.dumps({
            "N": N, "P": P, "K": K, "ph": pH,
            "temperature": temp, "humidity": humidity, "rainfall": rainfall
        }),
    )
    db.session.add(pred_row)   # INSERT — new row every time

    # FIX: always INSERT a new SoilData row — full audit trail per submission.
    soil_row = SoilData(
        user_id = session["user_id"],
        N=N, P=P, K=K, pH=pH
    )
    db.session.add(soil_row)   # INSERT — new row every time

    db.session.commit()

    plan  = get_smart_plan(top3[0]["crop"], session.get("location", ""), {})
    risks = assess_risks(temp, humidity, rainfall, top3[0]["crop"])

    # Enrich each prediction with a confidence label and XAI explanation
    for p in top3:
        p["confidence_label"] = confidence_label(p["probability"])
        xai = explain_recommendation(
            p["crop"], N, P, K, pH, temp, humidity, rainfall
        )
        p["reasons"]  = xai["reasons"]
        p["warnings"] = xai["warnings"]

    return jsonify({"predictions": top3, "plan": plan, "risks": risks})


@app.route("/api/weather", methods=["GET", "POST"])
@login_required
def api_weather():
    data = request.get_json() or {}
    # FIX: no hardcoded "Hyderabad, India" fallback.
    # Use the session location only; if that's also empty, return an error so
    # the frontend can ask the user to provide their location explicitly.
    location = (data.get("location") or session.get("location") or "").strip()
    if not location:
        return jsonify({"error": "No location provided"}), 400

    weather  = fetch_weather(location)
    forecast = fetch_forecast(location)

    if "error" not in weather:
        # FIX: always INSERT a new WeatherRecord — never overwrite the last one.
        wr = WeatherRecord(
            user_id     = session["user_id"],
            location    = location,
            temperature = weather["temperature"],
            humidity    = weather["humidity"],
            rainfall    = weather["rainfall"],
            description = weather["description"],
        )
        db.session.add(wr)   # INSERT — new row every time
        db.session.commit()

    return jsonify({"current": weather, "forecast": forecast})


@app.route("/api/risk", methods=["POST"])
@login_required
def api_risk():
    data  = request.get_json() or {}
    risks = assess_risks(
        float(data.get("temperature", 25)),
        float(data.get("humidity",    70)),
        float(data.get("rainfall",   100)),
        data.get("crop", ""),
    )
    return jsonify({"risks": risks})


@app.route("/api/plan", methods=["POST"])
@login_required
def api_plan():
    data = request.get_json() or {}
    crop = data.get("crop", "")
    plan = get_smart_plan(crop, session.get("location", ""), {})
    return jsonify({"plan": plan})


@app.route("/api/history")
@login_required
def api_history():
    preds = (
        Prediction.query
        .filter_by(user_id=session["user_id"])
        .order_by(Prediction.predicted_at.desc())
        .limit(20).all()
    )
    return jsonify([{
        "crop":        p.crop,
        "probability": p.probability,
        "date":        p.predicted_at.strftime("%Y-%m-%d %H:%M"),
        "inputs":      json.loads(p.inputs_json or "{}"),
    } for p in preds])


@app.route("/api/profile", methods=["POST"])
@login_required
def api_profile():
    data     = request.get_json() or {}
    location = data.get("location", "").strip()
    user     = db.session.get(User, session["user_id"])
    if not user:
        session.clear()
        return jsonify({"error": "Session expired"}), 401
    if location:
        # UPDATE the single "current location" column on the user row (correct)
        user.location = location
        db.session.commit()
        # FIX: also update the session so the new value is used immediately
        # and the old cached string is not served on subsequent requests
        session["location"] = location
    return jsonify({"status": "ok", "location": user.location})

# ── NEW: Forecast-based prediction endpoint ───────────────────────────────────
@app.route("/api/predict-forecast", methods=["POST"])
@login_required
def api_predict_forecast():
    """Use 5-day weather forecast averages instead of manually entered weather."""
    data = request.get_json() or {}
    try:
        N  = float(data["N"]);  P  = float(data["P"]);  K  = float(data["K"])
        pH = float(data["ph"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Soil inputs required: {e}"}), 400

    location = (data.get("location") or session.get("location") or "").strip()
    if not location:
        return jsonify({"error": "Location required for forecast prediction"}), 400

    forecast = fetch_forecast(location)
    if not forecast:
        return jsonify({"error": "Could not fetch forecast — check location or API key"}), 503

    top3 = predict_from_forecast(N, P, K, pH, forecast)
    for p in top3:
        p["confidence_label"] = confidence_label(p["probability"])
        xai = explain_recommendation(p["crop"], N, P, K, pH,
                                     mean([f["temperature"] for f in forecast[:40]]),
                                     mean([f["humidity"]    for f in forecast[:40]]),
                                     sum( [f["rainfall"]    for f in forecast[:40]]))
        p["reasons"]  = xai["reasons"]
        p["warnings"] = xai["warnings"]

    plan  = get_smart_plan(top3[0]["crop"], location, {})
    risks = assess_risks(mean([f["temperature"] for f in forecast[:40]]),
                         mean([f["humidity"]    for f in forecast[:40]]),
                         sum( [f["rainfall"]    for f in forecast[:40]]),
                         top3[0]["crop"])
    return jsonify({"predictions": top3, "plan": plan, "risks": risks,
                    "forecast_used": True, "forecast_days": len(set(f["datetime"][:10] for f in forecast))})


# ── NEW: Feedback submit ──────────────────────────────────────────────────────
@app.route("/api/feedback", methods=["POST"])
@login_required
def api_feedback():
    data = request.get_json() or {}
    fb = Feedback(
        user_id       = session["user_id"],
        prediction_id = data.get("prediction_id"),
        crop_planted  = data.get("crop_planted", "").strip(),
        rating        = int(data.get("rating", 3)),
        comment       = data.get("comment", "").strip(),
    )
    db.session.add(fb)
    db.session.commit()
    return jsonify({"status": "ok", "id": fb.id})


# ── NEW: Feedback list page ───────────────────────────────────────────────────
@app.route("/feedback")
@login_required
def feedback_page():
    user = db.session.get(User, session["user_id"])
    if not user:
        session.clear(); return redirect(url_for("login"))
    feedbacks = (Feedback.query.filter_by(user_id=user.id)
                 .order_by(Feedback.submitted_at.desc()).all())
    return render_template("feedback.html", feedbacks=feedbacks)


# ── NEW: Model comparison info ────────────────────────────────────────────────
@app.route("/api/model-info")
def api_model_info():
    return jsonify(model_meta)
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database tables ready.")
    app.run(debug=True, port=5000)
