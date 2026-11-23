# Corn Inventory — Streamlit Edition

A corn inventory analytics dashboard with the farm green and corn-gold theme.

## Deploy on Streamlit Community Cloud
1. Upload the contents of this folder to a GitHub repository (the `streamlit_app.py`, `requirements.txt`, `.streamlit/`, and `data/` folder must be at the repository root).
2. Visit https://share.streamlit.io and sign in.
3. Choose **Create app**.
4. Select your repository, branch (`main`), and main file path `streamlit_app.py`.
5. Click **Deploy**.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app uses the bundled `data/Corn_inventory_structured_numerical.csv`. Uploaded CSVs are held in the current Streamlit session and must contain `Sector`, `Year`, `Month`, and `Corn_Inventory`.

The classification models, 80/20 split, random_state=42, median-based High Inventory label, and SVR setup follow the current Corn Inventory Flask project's modeling workflow.
