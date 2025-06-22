import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("data_cleaned.csv")
# Encode country names
le = LabelEncoder()
df['country_encoded'] = le.fit_transform(df['country'])

# Define features and target
features = ['country_encoded', 'year', 'cereal_yield', 'fdi_perc_gdp', 'en_per_gdp',
            'en_per_cap', 'pop_urb_aggl_perc', 'prot_area_perc', 'gdp', 'gni_per_cap',
            'under_5_mort_rate', 'pop_growth_perc', 'pop', 'urb_pop_growth_perc', 'urb_pop']
target = 'co2_ttl'
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
importances = model.feature_importances_
feature_names = df[features].columns
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
import joblib
joblib.dump(model, "carbon_emission_model.pkl")

