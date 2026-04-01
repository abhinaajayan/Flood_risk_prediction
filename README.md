# Flood Risk Prediction System

A Flask-based web app that predicts flood risk using a pre-trained machine learning model. The app collects geospatial and environmental inputs, scales them, performs inference with a saved model, and returns a numeric risk score along with risk and livability categories.

## 📁 Project structure

- `app.py` - Flask application, routes, input handling, inference logic
- `templates/index.html` - User interface (form + result display)
- `best_model_FLOOD.pkl` - Serialized trained model
- `scaler_flood.pkl` - Serialized feature scaler
- `encoders_flood.joblib` - Optional encoder artifact (not used in `app.py`)

## 🛠️ Requirements

- Python 3.10+
- pip packages:
  - Flask
  - pandas
  - joblib
  - scikit-learn
  - numpy

Install dependencies:

```bash
pip install flask pandas joblib scikit-learn numpy
```

## ▶️ Run the app

From project root:

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

## 🧩 Input fields

Numeric fields:
- latitude
- longitude
- elevation_m
- distance_to_river_m
- population_density_per_km2
- built_up_percent
- rainfall_7d_mm
- monthly_rainfall_mm
- drainage_index
- ndvi
- ndwi
- historical_flood_count
- infrastructure_score
- inundation_area_sqm

Categorical fields:
- landcover: `Agriculture`, `Bare Soil`, `Forest`, `Plantation`, `Scrub`, `Urban`, `Wetland`
- soil_type: `Clay`, `Loamy`, `Peaty`, `Sandy`, `Silty`
- urban_rural: `Rural`, `Urban`

## 🧾 Output

- `prediction`: numeric flood risk score (rounded to 2 decimal places)
- `risk_level`: `Low Risk` (<30), `Medium Risk` (30-59), `High Risk` (>=60)
- `is_good_to_live`: `Yes` if score < 50, otherwise `No`

## 🛡️ Error handling

If validation or model steps fail, the app returns `Error: <message>` in place of numeric prediction.

## 💡 Notes

- Keep `best_model_FLOOD.pkl` and `scaler_flood.pkl` in the project root.
- For deployment, use a production WSGI server (`gunicorn`, `waitress`) and disable `debug=True`.
- Optionally add a `requirements.txt` by running `pip freeze > requirements.txt`.
