# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("best_model_FLOOD.pkl")
scaler = joblib.load("scaler_flood.pkl")

# Mapping dictionaries
landcover_map = {
    'Agriculture': 0, 'Bare Soil': 1, 'Forest': 2, 'Plantation': 3,
    'Scrub': 4, 'Urban': 5, 'Wetland': 6
}

soil_map = {
    'Clay': 0, 'Loamy': 1, 'Peaty': 2, 'Sandy': 3, 'Silty': 4
}

urban_rural_map = {
    'Rural': 0, 'Urban': 1
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    is_good_to_live = None
    risk_level = None
    input_values = {}

    if request.method == "POST":
        try:
            # Numeric inputs
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            elevation_m = float(request.form["elevation_m"])
            distance_to_river_m = float(request.form["distance_to_river_m"])
            population_density_per_km2 = float(request.form["population_density_per_km2"])
            built_up_percent = float(request.form["built_up_percent"])
            rainfall_7d_mm = float(request.form["rainfall_7d_mm"])
            monthly_rainfall_mm = float(request.form["monthly_rainfall_mm"])
            drainage_index = float(request.form["drainage_index"])
            ndvi = float(request.form["ndvi"])
            ndwi = float(request.form["ndwi"])
            historical_flood_count = float(request.form["historical_flood_count"])
            infrastructure_score = float(request.form["infrastructure_score"])
            inundation_area_sqm = float(request.form["inundation_area_sqm"])

            # Categorical inputs
            landcover = landcover_map[request.form["landcover"]]
            soil_type = soil_map[request.form["soil_type"]]
            urban_rural = urban_rural_map[request.form["urban_rural"]]

            # Create DataFrame (MATCH TRAINING FEATURES EXACTLY)
            X_input = pd.DataFrame([[
                latitude, longitude, elevation_m, distance_to_river_m,
                landcover, soil_type, population_density_per_km2,
                built_up_percent, urban_rural, rainfall_7d_mm,
                monthly_rainfall_mm, drainage_index, ndvi, ndwi,
                historical_flood_count, infrastructure_score,
                inundation_area_sqm
            ]], columns=[
                'latitude', 'longitude', 'elevation_m', 'distance_to_river_m',
                'landcover', 'soil_type', 'population_density_per_km2',
                'built_up_percent', 'urban_rural', 'rainfall_7d_mm',
                'monthly_rainfall_mm', 'drainage_index', 'ndvi', 'ndwi',
                'historical_flood_count', 'infrastructure_score',
                'inundation_area_sqm'
            ])

            # Scale
            X_scaled = scaler.transform(X_input)

            # Predict
            prediction = round(model.predict(X_scaled)[0], 2)

            # Risk category
            if prediction < 30:
                risk_level = "Low Risk"
            elif prediction < 60:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"

            # Livability
            is_good_to_live = "Yes" if prediction < 50 else "No"

            input_values = request.form

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        is_good_to_live=is_good_to_live,
        risk_level=risk_level,
        input_values=input_values,
        landcover_options=list(landcover_map.keys()),
        soil_options=list(soil_map.keys()),
        urban_rural_options=list(urban_rural_map.keys())
    )

if __name__ == "__main__":
    app.run(debug=True)