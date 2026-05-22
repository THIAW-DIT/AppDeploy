import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Exemple de données
X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
y = [2, 4, 6, 8, 10]

# Entraînement
model = LinearRegression()
model.fit(X, y)

# Sauvegarde du modèle
joblib.dump(model, "modele.pkl")
