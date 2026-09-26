import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_delivery_eda.csv"
df = pd.read_csv(DATA_PATH)


# 1. Supposons que ton DataFrame propre s'appelle 'df'
# Séparation des features (X) et de la cible (y)
X = df.drop(columns=['Time_taken(min)'])
y = df['Time_taken(min)']

# 2. Division des données (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Taille de l entraînement : {X_train.shape}')
print(f'Taille du test : {X_test.shape}')

# 3. Identification automatique des colonnes numériques et catégorielles
numeric_features = X.select_dtypes(
    include=['int64', 'float64', 'int32', 'float32']
).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 4. Création des transformateurs
# - Scaler pour les chiffres (StandardScaler)
# - Encoder pour le texte (OneHotEncoder)
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(
    handle_unknown='ignore', sparse_output=False
)

# Assemblage dans un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# 5. Définition des 4 modèles à tester et comparer
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (Régression régularisée)': Ridge(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
}

# 6. Entraînement et Évaluation de chaque modèle
results = []

for name, model in models.items():
  # Création d'un Pipeline pour lier le préprocessing et le modèle (évite le data leakage)
  pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

  # Entraînement sur les données d'entraînement
  pipeline.fit(X_train, y_train)

  # Prédiction sur les données de test
  y_pred = pipeline.predict(X_test)

  # Calcul des métriques de régression
  mae = mean_absolute_error(y_test, y_pred)  # Erreur absolue moyenne (en minutes)
  rmse = np.sqrt(
      mean_squared_error(y_test, y_pred)
  )  # Racine de l'erreur quadratique moyenne
  r2 = r2_score(y_test, y_pred)  # Coefficient de détermination

  results.append({'Modèle': name, 'MAE (min)': mae, 'RMSE (min)': rmse, 'R²': r2})

# Affichage des résultats sous forme de tableau propre
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
