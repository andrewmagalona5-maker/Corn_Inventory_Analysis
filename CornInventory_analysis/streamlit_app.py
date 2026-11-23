import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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

st.set_page_config(page_title="Corn Inventory", page_icon="🌽", layout="wide")

# Corn farm palette
DARK = "#234528"
GREEN = "#315B35"
YELLOW = "#D9A514"
BLUE = "#3478B8"
PURPLE = "#8B5CB8"
GRAY = "#9AA49C"
CREAM = "#FBF8EE"
SECTOR_NAMES = {"1": "Corn: Total Stock", "2": "Corn: Household Stock", "3": "Corn: Commercial Stock"}

st.markdown(f"""
<style>
.stApp {{background:{CREAM}; color:#29352A}}
[data-testid="stSidebar"] {{background:{DARK}}}
[data-testid="stSidebar"] * {{color:#F8F5EB !important}}
h1,h2,h3 {{color:{DARK}}}
div[data-testid="stMetric"] {{background:#fffdf7;border:1px solid #e8e2cf;padding:16px;border-radius:12px}}
.stButton>button {{background:{GREEN};color:white;border:0;border-radius:8px}}
.stDownloadButton>button {{background:{YELLOW};color:{DARK};border:0;border-radius:8px}}
div[data-testid="stDataFrame"] {{border:1px solid #e8e2cf;border-radius:10px}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def read_default():
    p = os.path.join(os.path.dirname(__file__), "data", "Corn_inventory_structured_numerical.csv")
    df = pd.read_csv(p)
    return clean_data(df)

def clean_data(df):
    df = df.copy()
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

@st.cache_resource
def fit_models(data_json):
    df = pd.read_json(io.StringIO(data_json), orient="records")
    X = df[["year", "month", "sector"]].copy()
    X["sector"] = X["sector"].astype(str)
    y = df["corn_inventory"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svr = Pipeline([("preprocessor", make_preprocessor()),
                    ("regressor", SVR(kernel="rbf", C=100000, epsilon=0.1))])
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, svr_pred)))
    r2 = float(r2_score(y_test, svr_pred))

    median_value = float(y.median())
    y_class = (y > median_value).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    }
    fitted, metrics, matrices, roc_data = {}, [], {}, {}
    for name, estimator in classifiers.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("classifier", estimator)])
        pipe.fit(Xc_train, yc_train)
        pred = pipe.predict(Xc_test)
        prob = pipe.predict_proba(Xc_test)[:, 1]
        auc = float(roc_auc_score(yc_test, prob))
        fitted[name] = pipe
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(yc_test, pred),
            "Precision": precision_score(yc_test, pred, zero_division=0),
            "Recall": recall_score(yc_test, pred, zero_division=0),
            "F1-score": f1_score(yc_test, pred, zero_division=0),
            "AUC": auc
        })
        matrices[name] = confusion_matrix(yc_test, pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(yc_test, prob)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
    return {"svr": svr, "rmse": rmse, "r2": r2, "median": median_value,
            "classifiers": fitted, "metrics": pd.DataFrame(metrics),
            "matrices": matrices, "roc": roc_data}

def chart_layout(fig, height=360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color=DARK,
                      margin=dict(l=20,r=20,t=45,b=20), legend_title_text="")
    return fig

st.sidebar.markdown("## 🌽 Corn Inventory")
st.sidebar.caption("Farm Inventory Analytics")
page = st.sidebar.radio("NAVIGATION", ["Dashboard", "Analytics", "Model Evaluation", "Forecast", "Records", "Upload Dataset"])

try:
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
    else:
        df = read_default()
    models = fit_models(df.to_json(orient="records"))
except Exception as exc:
    st.error(f"Unable to load or process the dataset: {exc}")
    st.stop()

st.title("🌽 Corn Inventory")
st.caption("Corn farm inventory monitoring, analytics, and predictive insights")

if page == "Dashboard":
    st.subheader("Inventory Overview")
    a,b,c,d = st.columns(4)
    a.metric("Total Inventory", f"{df.corn_inventory.sum():,.2f}")
    b.metric("Average Inventory", f"{df.corn_inventory.mean():,.2f}")
    c.metric("Total Records", f"{len(df):,}")
    d.metric("Latest Year", f"{int(df.year.max())}")
    st.divider()
    left,right = st.columns(2)
    yearly = df.groupby("year", as_index=False).corn_inventory.sum()
    fig = px.bar(yearly, x="year", y="corn_inventory", title="Inventory by Year",
                 color_discrete_sequence=[YELLOW], labels={"year":"Year","corn_inventory":"Corn Inventory"})
    left.plotly_chart(chart_layout(fig), use_container_width=True)
    sector = df.groupby("sector_name",as_index=False).corn_inventory.mean()
    fig2 = px.bar(sector,x="sector_name",y="corn_inventory",title="Average Inventory by Sector",
                  color="sector_name",color_discrete_sequence=[GREEN,YELLOW,"#8DA66A"])
    right.plotly_chart(chart_layout(fig2),use_container_width=True)
    st.info(f"Classification threshold (dataset median): **{models['median']:,.2f}**")

elif page == "Analytics":
    st.subheader("Descriptive Analytics")
    left,right=st.columns(2)
    sector = df.groupby("sector_name",as_index=False).corn_inventory.agg(["mean","min","max"]).reset_index()
    fig=px.bar(sector,x="sector_name",y="mean",title="Average Inventory by Sector",
               color_discrete_sequence=[GREEN],labels={"mean":"Average Inventory","sector_name":"Sector"})
    left.plotly_chart(chart_layout(fig),use_container_width=True)
    monthly=df.groupby("month",as_index=False).corn_inventory.mean()
    month_names={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month"]=monthly.month.map(month_names)
    fig2=px.line(monthly,x="Month",y="corn_inventory",markers=True,title="Average Inventory by Month",
                 color_discrete_sequence=[YELLOW],labels={"corn_inventory":"Average Inventory"})
    right.plotly_chart(chart_layout(fig2),use_container_width=True)
    st.subheader("Inventory Distribution")
    fig3=px.histogram(df,x="corn_inventory",nbins=25,color_discrete_sequence=[GREEN],title="Corn Inventory Frequency")
    st.plotly_chart(chart_layout(fig3),use_container_width=True)
    st.dataframe(sector.rename(columns={"sector_name":"Sector","mean":"Average","min":"Minimum","max":"Maximum"}),use_container_width=True,hide_index=True)

elif page == "Model Evaluation":
    st.subheader("Predictive Model Evaluation")
    st.caption("Classification target: High Inventory means above the dataset median. Train/test split: 80/20, random_state=42.")
    c1,c2=st.columns(2)
    c1.metric("SVR · RMSE",f"{models['rmse']:,.4f}")
    c2.metric("SVR · R² score",f"{models['r2']:.4f}")
    st.subheader("Classification Model Comparison")
    met=models["metrics"].copy()
    display=met.copy()
    for col in ["Accuracy","Precision","Recall","F1-score","AUC"]:
        display[col]=display[col].map(lambda v:f"{v*100:.2f}%" if col!="AUC" else f"{v:.4f}")
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.subheader("ROC Curves")
    colors={"Naive Bayes":YELLOW,"Logistic Regression":GREEN,"Decision Tree":BLUE,"Random Forest":PURPLE}
    legend_cols=st.columns(4)
    for col,(name,color) in zip(legend_cols,colors.items()):
        col.markdown(f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color}'></span> **{name}**",unsafe_allow_html=True)
    fig=go.Figure()
    for name,v in models["roc"].items():
        fig.add_trace(go.Scatter(x=v["fpr"],y=v["tpr"],mode="lines",name=f"{name} (AUC {v['auc']:.3f})",
                                 line=dict(color=colors[name],width=3)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random guessing",
                             line=dict(color=GRAY,width=2,dash="dash")))
    fig.update_layout(title="Receiver Operating Characteristic",xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]))
    st.plotly_chart(chart_layout(fig,430),use_container_width=True)
    st.subheader("Confusion Matrices")
    names=list(models["matrices"].keys())
    tabs=st.tabs(names)
    for tab,name in zip(tabs,names):
        cm=models["matrices"][name]
        with tab:
            cm_df=pd.DataFrame(cm,index=["Actual Low","Actual High"],columns=["Predicted Low","Predicted High"])
            st.dataframe(cm_df,use_container_width=True)

elif page == "Forecast":
    st.subheader("High Inventory Probability Forecast")
    st.caption("Random Forest classification probabilities for 2026–2030. These are class probabilities, not predicted inventory quantities.")
    rf=models["classifiers"]["Random Forest"]
    years=range(2026,2031)
    sectors=sorted(df.sector.unique().tolist(),key=lambda x:int(x))
    rows=[]
    for year in years:
        for sector in sectors:
            xfuture=pd.DataFrame([{"year":year,"month":month,"sector":str(sector)} for month in range(1,13)])
            probs=rf.predict_proba(xfuture)[:,1]
            classes=rf.predict(xfuture)
            rows.append({"Year":year,"Sector":SECTOR_NAMES.get(str(sector),str(sector)),
                         "High inventory probability":float(np.mean(probs)),
                         "Months classified high":int(np.sum(classes==1))})
    forecast=pd.DataFrame(rows)
    fig=px.line(forecast,x="Year",y="High inventory probability",color="Sector",markers=True,
                title="Forecasted High Inventory Probability by Sector",
                color_discrete_sequence=[GREEN,YELLOW,BLUE])
    st.plotly_chart(chart_layout(fig),use_container_width=True)
    st.dataframe(forecast.style.format({"High inventory probability":"{:.1%}"}),use_container_width=True,hide_index=True)

elif page == "Records":
    st.subheader("Inventory Records")
    query=st.text_input("Search records",placeholder="Search sector, year, month, or inventory...")
    shown=df.copy()
    if query:
        mask=shown.astype(str).apply(lambda col:col.str.contains(query,case=False,na=False)).any(axis=1)
        shown=shown[mask]
    st.caption(f"{len(shown):,} matching records")
    st.dataframe(shown[["sector_name","year","month","corn_inventory"]].rename(
        columns={"sector_name":"Sector","year":"Year","month":"Month","corn_inventory":"Corn Inventory"}),
        use_container_width=True,hide_index=True)
    st.download_button("Download current dataset",df.to_csv(index=False).encode("utf-8"),
                       file_name="corn_inventory_dataset.csv",mime="text/csv")

elif page == "Upload Dataset":
    st.subheader("Upload a Replacement Dataset")
    st.write("CSV must contain these columns: Sector, Year, Month, Corn_Inventory.")
    uploaded=st.file_uploader("Choose CSV file",type=["csv"])
    if uploaded is not None:
        try:
            incoming=clean_data(pd.read_csv(uploaded))
            st.success(f"Valid dataset: {len(incoming):,} rows.")
            st.dataframe(incoming.head(10),use_container_width=True)
            if st.button("Use this dataset for this session"):
                st.session_state.uploaded_df=incoming
                st.cache_resource.clear()
                st.rerun()
        except Exception as exc:
            st.error(str(exc))
