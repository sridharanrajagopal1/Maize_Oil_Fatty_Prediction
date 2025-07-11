from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

# ---------- STEP 1: Select Features (SNPs) and Target (Oil) ----------
# Drop non-numeric + non-SNP columns
X = full_data.drop(columns=["Env", "Rep", "Genotype", "Oil", "C16:0", "C16:1", "C18:0", "C18:1", "C18:2",
                            "C18:3", "C20:0", "C20:1", "C22:0", "C24:0"], errors='ignore')

# Target variable
y = full_data["Oil"]

# ---------- STEP 2: Split Train/Test ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- STEP 3: Train Ridge Regression ----------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ---------- STEP 4: Predict ----------
y_pred = model.predict(X_test)

# ---------- STEP 5: Evaluate ----------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
corr, _ = pearsonr(y_test, y_pred)

print("📊 Evaluation Metrics:")
print("✅ RMSE:", round(rmse, 4))
print("✅ R² Score:", round(r2, 4))
print("✅ Pearson Correlation:", round(corr, 4))

# ---------- STEP 6: Plot Observed vs Predicted ----------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Observed Oil Content")
plt.ylabel("Predicted Oil Content")
plt.title("Observed vs Predicted (Ridge Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()
