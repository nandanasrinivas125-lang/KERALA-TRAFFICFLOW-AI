from flask import Flask, render_template, request
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)


# ==================================================
# LOAD DATASET
# ==================================================

data = pd.read_csv(
    r"C:\Users\Nanda\Downloads\road_network.csv"
)


# ==================================================
# DATE PROCESSING
# ==================================================

data["last_maintenance_date"] = pd.to_datetime(
    data["last_maintenance_date"]
)

data["maintenance_year"] = (
    data["last_maintenance_date"].dt.year
)

data["maintenance_month"] = (
    data["last_maintenance_date"].dt.month
)

latest_date = data["last_maintenance_date"].max()

data["maintenance_age_days"] = (
    latest_date - data["last_maintenance_date"]
).dt.days


# ==================================================
# LABEL ENCODING
# ==================================================

road_encoder = LabelEncoder()
surface_encoder = LabelEncoder()
district_encoder = LabelEncoder()


data["road_type"] = road_encoder.fit_transform(
    data["road_type"]
)

data["surface_condition"] = surface_encoder.fit_transform(
    data["surface_condition"]
)

data["district"] = district_encoder.fit_transform(
    data["district"]
)


# ==================================================
# DROP UNNECESSARY COLUMNS
# ==================================================

data = data.drop(
    columns=[
        "road_id",
        "road_name",
        "last_maintenance_date"
    ]
)


# ==================================================
# INPUT AND TARGET
# ==================================================

x = data.drop(
    ["avg_daily_traffic"],
    axis=1
)

y = data["avg_daily_traffic"]


# ==================================================
# TRAIN MODEL
# ==================================================

model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

model.fit(x, y)


# ==================================================
# KERALA LOCATIONS
# ==================================================

kerala_places = [
    "Thiruvananthapuram",
    "Kollam",
    "Pathanamthitta",
    "Alappuzha",
    "Kottayam",
    "Idukki",
    "Ernakulam",
    "Thrissur",
    "Palakkad",
    "Malappuram",
    "Kozhikode",
    "Wayanad",
    "Kannur",
    "Kasaragod",
    "Kochi",
    "Aluva",
    "Kakkanad",
    "Thrippunithura",
    "Perumbavoor",
    "Muvattupuzha",
    "Changanassery",
    "Pala",
    "Cherthala",
    "Kayamkulam",
    "Varkala",
    "Attingal",
    "Chavakkad",
    "Ottapalam",
    "Manjeri",
    "Tirur",
    "Vadakara",
    "Kalpetta",
    "Thalassery"
]


# ==================================================
# APPROXIMATE LOCATION VALUES
# ==================================================

location_values = {

    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kollam": (8.8932, 76.6141),
    "Pathanamthitta": (9.2648, 76.7870),
    "Alappuzha": (9.4981, 76.3388),
    "Kottayam": (9.5916, 76.5222),
    "Idukki": (9.8494, 76.9720),
    "Ernakulam": (9.9816, 76.2999),
    "Thrissur": (10.5276, 76.2144),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0510, 76.0711),
    "Kozhikode": (11.2588, 75.7804),
    "Wayanad": (11.6854, 76.1320),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.4996, 74.9869),

    "Kochi": (9.9312, 76.2673),
    "Aluva": (10.1076, 76.3516),
    "Kakkanad": (10.0159, 76.3419),
    "Thrippunithura": (9.9495, 76.3495),
    "Perumbavoor": (10.1060, 76.4730),
    "Muvattupuzha": (9.9849, 76.5773),
    "Changanassery": (9.4429, 76.5360),
    "Pala": (9.7167, 76.6844),
    "Cherthala": (9.6850, 76.3350),
    "Kayamkulam": (9.1817, 76.5009),
    "Varkala": (8.7379, 76.7163),
    "Attingal": (8.6961, 76.8151),
    "Chavakkad": (10.5285, 76.0476),
    "Ottapalam": (10.7725, 76.3776),
    "Manjeri": (11.1201, 76.1190),
    "Tirur": (10.9137, 75.9218),
    "Vadakara": (11.6016, 75.5917),
    "Kalpetta": (11.6085, 76.0830),
    "Thalassery": (11.7480, 75.4894)
}


# ==================================================
# KERALA ROAD TYPES
# ==================================================

# These are displayed to the user.
# They are mapped internally to the road types
# available in your existing dataset.

road_type_map = {

    "National Highway":
        "Jalan Arteri",

    "State Highway":
        "Jalan Arteri",

    "Major District Road":
        "Jalan Kolektor",

    "Other District Road":
        "Jalan Lokal",

    "Local Road":
        "Jalan Lingkungan",

    "Toll Road":
        "Jalan Tol"
}


# ==================================================
# ROAD SURFACE
# ==================================================

surface_map = {

    "Good":
        "Baik",

    "Average":
        "Cukup",

    "Poor":
        "Rusak Ringan",

    "Very Poor":
        "Rusak Berat"
}


# ==================================================
# HOME
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            # ------------------------------------------
            # USER INPUT
            # ------------------------------------------

            location = request.form["location"]

            road_type_display = request.form[
                "road_type"
            ]

            road_length = float(
                request.form["road_length"]
            )

            lanes = int(
                request.form["lanes"]
            )

            speed_limit = int(
                request.form["speed_limit"]
            )

            bus_lane = int(
                request.form["bus_lane"]
            )

            bike_lane = int(
                request.form["bike_lane"]
            )

            surface_display = request.form[
                "surface_condition"
            ]


            # ------------------------------------------
            # ROAD TYPE
            # ------------------------------------------

            road_type_original = road_type_map[
                road_type_display
            ]

            road_type_encoded = (
                road_encoder.transform(
                    [road_type_original]
                )[0]
            )


            # ------------------------------------------
            # SURFACE CONDITION
            # ------------------------------------------

            surface_original = surface_map[
                surface_display
            ]

            surface_encoded = (
                surface_encoder.transform(
                    [surface_original]
                )[0]
            )


            # ------------------------------------------
            # LOCATION
            # ------------------------------------------

            latitude_start, longitude_start = (
                location_values[location]
            )

            latitude_end = (
                latitude_start + 0.01
            )

            longitude_end = (
                longitude_start + 0.01
            )


            # ------------------------------------------
            # AUTOMATIC VALUES
            # ------------------------------------------

            # Average capacity from dataset
            road_capacity = int(
                data[
                    "capacity_vehicles_per_hour"
                ].mean()
            )


            # Median maintenance information
            maintenance_age_days = int(
                data[
                    "maintenance_age_days"
                ].median()
            )

            maintenance_year = int(
                data[
                    "maintenance_year"
                ].median()
            )

            maintenance_month = int(
                data[
                    "maintenance_month"
                ].median()
            )


            # No toll by default
            toll_rate = 0


            # Most common district
            common_district = (
                data["district"].mode()[0]
            )


            # ------------------------------------------
            # INPUT DATA
            # ------------------------------------------

            input_data = pd.DataFrame({

                "road_type": [
                    road_type_encoded
                ],

                "length_km": [
                    road_length
                ],

                "lanes": [
                    lanes
                ],

                "speed_limit_kmh": [
                    speed_limit
                ],

                "capacity_vehicles_per_hour": [
                    road_capacity
                ],

                "has_bus_lane": [
                    bus_lane
                ],

                "has_bike_lane": [
                    bike_lane
                ],

                "surface_condition": [
                    surface_encoded
                ],

                "latitude_start": [
                    latitude_start
                ],

                "longitude_start": [
                    longitude_start
                ],

                "latitude_end": [
                    latitude_end
                ],

                "longitude_end": [
                    longitude_end
                ],

                "district": [
                    common_district
                ],

                "toll_rate_rp": [
                    toll_rate
                ],

                "maintenance_year": [
                    maintenance_year
                ],

                "maintenance_month": [
                    maintenance_month
                ],

                "maintenance_age_days": [
                    maintenance_age_days
                ]
            })


            # ------------------------------------------
            # PREDICTION
            # ------------------------------------------

            result = model.predict(
                input_data
            )[0]

            prediction = round(result)


        except Exception as e:

            error = str(e)


    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        kerala_places=kerala_places
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(debug=True)
