from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import os, io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)

app = Flask(__name__)
app.secret_key = "cornsight-local-development-key"
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DEFAULT_CSV = os.path.join(DATA_DIR, "Corn_inventory_structured_numerical.csv")
ACTIVE_CSV = os.path.join(DATA_DIR, "active_inventory.csv")
SECTOR_NAMES = {"1": "Corn: Total Stock", "2": "Corn: Household Stock", "3": "Corn: Commercial Stock"}

def load_data():
    path = ACTIVE_CSV if os.path.exists(ACTIVE_CSV) else DEFAULT_CSV
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    required = {"sector", "year", "month", "corn_inventory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    for col in ["sector", "year", "month", "corn_inventory"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required)).copy()
    df["sector"] = df["sector"].astype(int).astype(str)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["sector_name"] = df["sector"].map(SECTOR_NAMES).fillna(df["sector"])
    return df

def make_preprocessor():
    try:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", StandardScaler(), ["year", "month"]),
        ("cat", encoder, ["sector"])
    ])

def train_models(df):
    X = df[["year", "month", "sector"]].copy()
    X["sector"] = X["sector"].astype(str)
    y = df["corn_inventory"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    prep = make_preprocessor()
    svr = Pipeline([("preprocessor", prep),
                    ("regressor", SVR(kernel="rbf", C=100000, epsilon=0.1))])
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, svr_pred)))
    r2 = float(r2_score(y_test, svr_pred))

    median_value = float(y.median())
    y_class = (y > median_value).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    }
    fitted, metrics, matrices, roc = {}, [], {}, {}
    for name, estimator in classifiers.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("classifier", estimator)])
        pipe.fit(Xc_train, yc_train)
        pred = pipe.predict(Xc_test)
        prob = pipe.predict_proba(Xc_test)[:, 1]
        fitted[name] = pipe
        metrics.append({
            "model": name,
            "accuracy": float(accuracy_score(yc_test, pred)),
            "precision": float(precision_score(yc_test, pred, zero_division=0)),
            "recall": float(recall_score(yc_test, pred, zero_division=0)),
            "f1": float(f1_score(yc_test, pred, zero_division=0)),
            "auc": float(roc_auc_score(yc_test, prob))
        })
        cm = confusion_matrix(yc_test, pred, labels=[0, 1])
        matrices[name] = cm.tolist()
        fpr, tpr, _ = roc_curve(yc_test, prob)
        roc[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                     "auc": float(roc_auc_score(yc_test, prob))}
    return {
        "svr": svr, "svr_rmse": rmse, "svr_r2": r2,
        "median": median_value, "classifiers": fitted, "metrics": metrics,
        "matrices": matrices, "roc": roc, "test_actual": y_test.tolist(),
        "test_predicted": svr_pred.tolist()
    }

def get_context():
    df = load_data()
    models = train_models(df)
    return df, models

def fmt(v):
    return f"{float(v):,.2f}"

@app.route("/")
def dashboard():
    try:
        df, m = get_context()
        total = float(df.corn_inventory.sum())
        average = float(df.corn_inventory.mean())
        latest_year = int(df.year.max())
        yearly = df.groupby("year").corn_inventory.sum().reset_index()
        sectors = df.groupby("sector_name").corn_inventory.mean().reset_index()
        monthly = df.groupby("month").corn_inventory.mean().reindex(range(1,13), fill_value=0)
        return render_template("dashboard.html", total=fmt(total), average=fmt(average),
            records=len(df), latest_year=latest_year, years=yearly.year.tolist(),
            yearly_values=yearly.corn_inventory.tolist(),
            sector_labels=sectors.sector_name.tolist(), sector_values=sectors.corn_inventory.tolist(),
            month_labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            month_values=monthly.tolist(), median=fmt(m["median"]))
    except Exception as e:
        return render_template("error.html", error=str(e)), 500

@app.route("/analytics")
def analytics():
    df, _ = get_context()
    distribution = df.groupby("sector_name").corn_inventory.agg(["mean","min","max"]).reset_index()
    return render_template("analytics.html", labels=distribution.sector_name.tolist(),
        means=distribution["mean"].tolist(), mins=distribution["min"].tolist(),
        maxs=distribution["max"].tolist(), values=df.corn_inventory.tolist(),
        years=sorted(df.year.unique().tolist()))

@app.route("/models")
def models_page():
    _, m = get_context()
    return render_template("models.html", metrics=m["metrics"], rmse=fmt(m["svr_rmse"]),
        r2=f'{m["svr_r2"]:.4f}', matrices=m["matrices"], roc=m["roc"],
        roc_json=__import__("json").dumps(m["roc"]), matrices_json=__import__("json").dumps(m["matrices"]))

@app.route("/forecast")
def forecast():
    df, m = get_context()
    years = list(range(2026, 2031))
    sectors = sorted(df.sector.unique().tolist(), key=lambda x: int(x))
    records = []
    rf = m["classifiers"]["Random Forest"]
    for year in years:
        for sector in sectors:
            features = pd.DataFrame([{"year": year, "month": month, "sector": str(sector)}
                                     for month in range(1, 13)])
            probabilities = rf.predict_proba(features)[:, 1]
            classes = rf.predict(features)
            records.append({
                "year": year, "sector": SECTOR_NAMES.get(str(sector), str(sector)),
                "probability": float(np.mean(probabilities)),
                "high_months": int(np.sum(classes == 1))
            })
    return render_template("forecast.html", forecast=records,
        years=years, labels=list(dict.fromkeys(x["sector"] for x in records)),
        values=[[next(x["probability"] for x in records if x["year"] == y and x["sector"] == s)
                 for y in years] for s in list(dict.fromkeys(x["sector"] for x in records))])

@app.route("/records")
def records():
    df = load_data()
    q = request.args.get("q", "").strip()
    if q:
        mask = (df.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False))).any(axis=1)
        df = df[mask]
    page = max(1, int(request.args.get("page", 1)))
    per_page = 20
    total = len(df)
    pages = max(1, (total + per_page - 1)//per_page)
    page = min(page, pages)
    subset = df.iloc[(page-1)*per_page:page*per_page]
    rows = subset[["sector_name","year","month","corn_inventory"]].to_dict("records")
    return render_template("records.html", rows=rows, q=q, page=page, pages=pages, total=total)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("upload"))
        try:
            incoming = pd.read_csv(file)
            incoming.columns = incoming.columns.str.strip().str.lower()
            required = {"sector", "year", "month", "corn_inventory"}
            if not required.issubset(incoming.columns):
                raise ValueError("Required columns: Sector, Year, Month, Corn_Inventory.")
            incoming.to_csv(ACTIVE_CSV, index=False)
            flash(f"Dataset uploaded successfully ({len(incoming):,} rows).", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            flash(str(e), "error")
    return render_template("upload.html")

@app.route("/download")
def download():
    path = ACTIVE_CSV if os.path.exists(ACTIVE_CSV) else DEFAULT_CSV
    return send_file(path, as_attachment=True, download_name="corn_inventory_dataset.csv")

@app.route("/api/overview")
def api_overview():
    df, m = get_context()
    return jsonify({"records": len(df), "median_threshold": m["median"],
                    "models": m["metrics"], "svr_rmse": m["svr_rmse"], "svr_r2": m["svr_r2"]})

if __name__ == "__main__":
    app.run(debug=True)
