# Corn Inventory — Corn Inventory Analytics Website

A local Flask dashboard built around the supplied structured numerical dataset and the workflow in the supplied Colab notebook.

## Included dataset
`data/Corn_inventory_structured_numerical.csv` (936 records; columns: Sector, Year, Month, Corn_Inventory).

Sector codes follow the notebook mapping:
- 1 = Corn: Total Stock
- 2 = Corn: Household Stock
- 3 = Corn: Commercial Stock

## Requirements
- Python 3.11–3.13 recommended for broad scientific-package compatibility. If using a newer Python and scikit-learn has no compatible wheel yet, install Python 3.12 or 3.13.
- Packages are listed in `requirements.txt`.

## Run on Windows PowerShell
Open PowerShell in the folder containing `app.py`, then:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Keep the terminal open.

## Pages
- Dashboard: total/average inventory, yearly trend, sector averages, monthly profile
- Data Analytics: sector averages/ranges and inventory distribution
- Model Evaluation: SVR RMSE/R²; Naive Bayes, Logistic Regression, Decision Tree, Random Forest metrics; ROC/AUC and confusion matrices
- Forecast: Random Forest probability of High Inventory by sector for 2026–2030
- Inventory Records: searchable, paginated table
- Upload Dataset: replace active CSV (requires Sector, Year, Month, Corn_Inventory)
- Download CSV: download active dataset

## Methodology reflected from the Colab
- Regression target: `Corn_Inventory`; features: `Year`, `Month`, `Sector`
- 80/20 train-test split, random_state=42
- Numeric Year and Month standardized; Sector one-hot encoded
- SVR: RBF kernel, C=100000, epsilon=0.1
- Classification target: 1 if inventory is above the full dataset median, otherwise 0
- Classifiers: Gaussian Naive Bayes, Logistic Regression, Decision Tree (max_depth=5), Random Forest (100 estimators, max_depth=5)
- Forecast: monthly future inputs for 2026–2030; Random Forest class and high-inventory probability summarized by sector/year

Note: the notebook's future section predicts classification status/probability, not a numeric inventory quantity. This site labels it accordingly. Results are retrained locally from the included CSV when pages load. The Google Fonts and Chart.js CDN need internet access for remote fonts and charts.

## Visual theme
The interface uses a restrained sage-green and corn-yellow palette with warm off-white surfaces, muted olive accents, and high-contrast text for a simple professional agricultural analytics look.
