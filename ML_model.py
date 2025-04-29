# Fertilizer Prediction ML Model - Improved Version

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load Data
df = pd.read_csv("agriculture_dataset_500.csv")

# Basic Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)  # remove missing values

# Encode Categorical Features
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

# Features and Target
X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

# Train/Test Split (Stratify to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tune Random Forest with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Model Accuracy: {accuracy*100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance (Optional: drop useless features)
importances = pd.Series(best_model.feature_importances_, index=X.columns)
important_features = importances[importances > 0.01].index.tolist()
print("Important Features:", important_features)

# Re-train using only important features (Optional)
X_train_imp = X_train[important_features]
X_test_imp = X_test[important_features]

best_model.fit(X_train_imp, y_train)
y_pred_imp = best_model.predict(X_test_imp)
accuracy_imp = accuracy_score(y_test, y_pred_imp)
print(f"Accuracy after Feature Selection: {accuracy_imp*100:.2f}%")

# Save Model and Label Encoders
joblib.dump(best_model, 'fertilizer_model.pkl')
joblib.dump(le_soil, 'soil_encoder.pkl')
joblib.dump(le_crop, 'crop_encoder.pkl')
joblib.dump(le_fert, 'fertilizer_encoder.pkl')

# Example Prediction
# new_data = pd.DataFrame({
#     'Temperature': [25],
#     'Humidity': [60],
#     'Moisture': [30],
#     'Soil Type': le_soil.transform(['Clay']),
#     'Crop Type': le_crop.transform(['Rice']),
#     'Nitrogen': [50],
#     'Potassium': [40],
#     'Phosphorous': [30]
# })
# new_data = new_data[important_features]  # Keep only important features
# prediction = best_model.predict(new_data)
# fertilizer_name = le_fert.inverse_transform(prediction)
# print(f"Recommended Fertilizer: {fertilizer_name[0]}")
