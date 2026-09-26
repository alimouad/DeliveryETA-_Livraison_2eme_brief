from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_delivery_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_delivery_eda.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "final_model_metrics.csv"
MODEL_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_results.csv"

st.set_page_config(
    page_title="Delivery ETA",
    page_icon="⌛",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_model_results():
    return pd.read_csv(MODEL_RESULTS_PATH)


def haversine_distance(lat1, lon1, lat2, lon2):
    from math import asin, cos, radians, sin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    )
    return 6371 * 2 * asin(sqrt(value))


def build_features(
    distance_km,
    restaurant_latitude,
    restaurant_longitude,
    delivery_latitude,
    delivery_longitude,
    weather,
    traffic,
    vehicle,
    pickup_hour,
    pickup_minute,
):
    traffic_score = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}[traffic]
    peak_hour = int(pickup_hour in {8, 9, 13, 14, 19, 20, 21})
    distance_traffic = distance_km * traffic_score

    return pd.DataFrame(
        [
            {
                "Restaurant_latitude": restaurant_latitude,
                "Restaurant_longitude": restaurant_longitude,
                "Delivery_location_latitude": delivery_latitude,
                "Delivery_location_longitude": delivery_longitude,
                "Weatherconditions": weather,
                "Road_traffic_density": traffic,
                "Type_of_vehicle": vehicle,
                "distance_km": distance_km,
                "pickup_hour": pickup_hour,
                "pickup_minute": pickup_minute,
                "is_peak_hour": peak_hour,
                "Traffic_Score": traffic_score,
                "Distance_Traffic": distance_traffic,
            }
        ]
    )


st.title("Delivery ETA")
st.caption("Prédiction du temps de livraison en minutes")

try:
    model = load_model()
    data = load_data()
    final_metrics = load_metrics()
    model_results = load_model_results()
except FileNotFoundError as error:
    st.error(f"Fichier nécessaire introuvable : {error.filename}")
    st.stop()

with st.sidebar:
    st.header("Informations de la commande")
    input_mode = st.radio("Distance", ["Distance directe", "Coordonnées"])

    if input_mode == "Distance directe":
        distance_km = st.number_input(
            "Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1
        )
        restaurant_latitude = 19.0
        restaurant_longitude = 76.6
        delivery_latitude = restaurant_latitude
        delivery_longitude = restaurant_longitude
    else:
        restaurant_latitude = st.number_input(
            "Latitude restaurant", min_value=-90.0, max_value=90.0, value=19.0
        )
        restaurant_longitude = st.number_input(
            "Longitude restaurant", min_value=-180.0, max_value=180.0, value=76.6
        )
        delivery_latitude = st.number_input(
            "Latitude livraison", min_value=-90.0, max_value=90.0, value=19.05
        )
        delivery_longitude = st.number_input(
            "Longitude livraison", min_value=-180.0, max_value=180.0, value=76.65
        )
        distance_km = haversine_distance(
            restaurant_latitude,
            restaurant_longitude,
            delivery_latitude,
            delivery_longitude,
        )
        st.caption(f"Distance calculée : {distance_km:.2f} km")

    weather = st.selectbox(
        "Météo",
        ["Sunny", "Cloudy", "Rainy", "Stormy", "Windy", "Foggy", "Sandstorms"],
    )
    traffic = st.selectbox("Trafic", ["Low", "Medium", "High", "Jam"])
    vehicle = st.selectbox("Type de véhicule", ["motorcycle", "scooter", "electric_scooter", "bicycle", "car"])
    courier_rating = st.slider("Note du livreur", 1.0, 5.0, 4.5, 0.1)
    pickup_hour = st.slider("Heure de collecte", 0, 23, 19)
    pickup_minute = st.select_slider("Minute", options=list(range(0, 60, 5)), value=30)

    predict_clicked = st.button("Prédire le temps", type="primary", use_container_width=True)

if predict_clicked:
    features = build_features(
        distance_km,
        restaurant_latitude,
        restaurant_longitude,
        delivery_latitude,
        delivery_longitude,
        weather,
        traffic,
        vehicle,
        pickup_hour,
        pickup_minute,
    )
    prediction = float(model.predict(features)[0])
    st.success(f"Temps de livraison estimé : {prediction:.0f} minutes")
    st.info(
        f"La note du livreur saisie est {courier_rating:.1f}/5. "
        "Elle est affichée, mais le modèle actuel n'a pas été entraîné avec cette variable."
    )

st.divider()
tab_data, tab_metrics, tab_accuracy = st.tabs(
    ["Visualisation des données", "Métriques du modèle", "Précision du modèle"]
)

with tab_data:
    st.subheader("Aperçu des données utilisées")
    st.dataframe(data.head(20), use_container_width=True)
    st.metric("Nombre de commandes", f"{len(data):,}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=data, x="Time_taken(min)", bins=25, kde=True, ax=ax)
        ax.set_title("Distribution du temps de livraison")
        ax.set_xlabel("Minutes")
        st.pyplot(fig, clear_figure=True)
    with col2:
        traffic_means = data.groupby("Road_traffic_density")["Time_taken(min)"].mean().sort_values()
        st.bar_chart(traffic_means, x_label="Trafic", y_label="Minutes moyennes")

with tab_metrics:
    st.subheader("Performances sur entraînement et test")
    st.dataframe(final_metrics, use_container_width=True)
    st.caption("MAE et RMSE sont exprimées en minutes. Un score R² plus proche de 1 est meilleur.")

    best_row = final_metrics.loc[final_metrics["Ensemble"] == "Test"].iloc[0]
    st.metric("Erreur moyenne sur test", f"{best_row['MAE (min)']:.2f} minutes")
    st.metric("RMSE sur test", f"{best_row['RMSE (min)']:.2f} minutes")
    st.metric("R² sur test", f"{best_row['R²']:.3f}")

with tab_accuracy:
    st.subheader("Comparaison des modèles")
    chart_data = model_results.set_index("Modèle")[["MAE (min)", "RMSE (min)"]]
    st.bar_chart(chart_data, y_label="Erreur en minutes")
    st.dataframe(model_results.round(4), use_container_width=True)
    st.write(
        "Le Gradient Boosting optimisé est le modèle final. "
        "Sur les commandes de test, il se trompe en moyenne d'environ "
        f"{best_row['MAE (min)']:.2f} minutes."
    )
