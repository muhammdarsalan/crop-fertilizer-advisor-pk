import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create models folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# ---------------- Crop Recommendation Model ----------------

print("Training crop model...")

crop_data = pd.read_csv("data/crop_recommendation.csv")

X = crop_data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = crop_data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(X_train, y_train)

crop_predictions = crop_model.predict(X_test)
crop_accuracy = accuracy_score(y_test, crop_predictions)

print("Crop Model Accuracy:", round(crop_accuracy * 100, 2), "%")

with open("models/crop_model.pkl", "wb") as file:
    pickle.dump(crop_model, file)

# ---------------- Fertilizer Recommendation Model ----------------

print("\nTraining fertilizer model...")

fert_data = pd.read_csv("data/fertilizer_recommendation.csv")

soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

fert_data["Soil Type"] = soil_encoder.fit_transform(fert_data["Soil Type"])
fert_data["Crop Type"] = crop_encoder.fit_transform(fert_data["Crop Type"])
fert_data["Fertilizer Name"] = fertilizer_encoder.fit_transform(
    fert_data["Fertilizer Name"]
)

X = fert_data[
    [
        "Temparature",
        "Humidity ",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
    ]
]

y = fert_data["Fertilizer Name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

fert_model = RandomForestClassifier(random_state=42)
fert_model.fit(X_train, y_train)

fert_predictions = fert_model.predict(X_test)
fert_accuracy = accuracy_score(y_test, fert_predictions)

print("Fertilizer Model Accuracy:", round(fert_accuracy * 100, 2), "%")

with open("models/fertilizer_model.pkl", "wb") as file:
    pickle.dump(fert_model, file)

encoders = {
    "soil": soil_encoder,
    "crop": crop_encoder,
    "fertilizer": fertilizer_encoder
}

with open("models/label_encoders.pkl", "wb") as file:
    pickle.dump(encoders, file)

print("\nModels saved successfully.")