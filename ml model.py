

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier, plot_importance

# 1. Load dataset
df = pd.read_csv("Human_Resources_Cleaned.csv")


# 2. Features and Target
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Calculate scale_pos_weight (imbalance fixing)
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# 6. Train XGBoost with class weight
xgb_model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
xgb_model.fit(X_train_scaled, y_train)

# 7. Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# 8. Evaluate Model
print("🔹 Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")



# 10. Feature Importance
plt.figure(figsize=(10,8))
plot_importance(xgb_model, max_num_features=10, height=0.5)
plt.title("Top 10 Important Features")
plt.show()

print("✅ Model ready for predicting employee attrition!")


# Save model
xgb_model.save_model('xgb_hr_model.json')

# Save scaler
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb')


















































