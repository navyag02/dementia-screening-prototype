import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Simulate the same features as extract_features
# 13 MFCCs + 1 pause feature = 14 features
np.random.seed(42)
X = np.random.rand(100, 14)  # <-- 14 features now
y = np.random.choice(["Healthy", "Dementia"], size=100)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "mock_hybrid_model.pkl")
print("✅ Model saved as mock_hybrid_model.pkl with 14 features")
