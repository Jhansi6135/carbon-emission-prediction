
# 🌍 Carbon Emission Prediction

This project uses machine learning to predict **country-level CO₂ emissions** based on environmental, economic, and demographic indicators. It was built using a dataset provided via the AICTE LMS portal (based on World Bank data).

---

## 📌 Overview

- **Objective**: Predict total CO₂ emissions (`co2_ttl`) using features like GDP, population, energy usage, urbanization, etc.
- **Type**: Regression Problem
- **Model Used**: Random Forest Regressor

---

## 🗃️ Dataset

- Filename: `data_cleaned.csv`
- Features include:
  - `country`, `year`
  - `gdp`, `gni_per_cap`, `fdi_perc_gdp`
  - `pop`, `urb_pop`, `pop_growth_perc`, etc.

> The dataset was cleaned and preprocessed prior to modeling.

---

## 🧠 ML Pipeline

1. **Label Encoding**: For country names
2. **Feature Scaling**: Using `StandardScaler`
3. **Modeling**: RandomForestRegressor from `sklearn`
4. **Evaluation**: MSE and R² Score
5. **Visualization**: Feature importance using Seaborn
6. **Model Saving**: Stored as `carbon_emission_model.pkl`

---

## 📂 Project Files

| File Name                    | Description                                      |
|-----------------------------|--------------------------------------------------|
| `carbon_emission_prediction.py` | Main Python script (can run in IDLE/VS Code)   |
| `data_cleaned.csv`          | Cleaned dataset (optional to upload)            |
| `carbon_emission_model.pkl` | Trained model saved using `joblib`              |
| `requirements.txt`          | List of Python dependencies                     |
| `README.md`                 | This project overview file                      |

---

## ▶️ How to Run

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main Python script:
   ```bash
   python carbon_emission_prediction.py
   ```

---

## 📊 Output

- Prints MSE and R² score for prediction performance
- Displays a bar plot of feature importance
- Saves the trained model as `carbon_emission_model.pkl`

---

## 🔮 Future Enhancements

- Use XGBoost or LightGBM for better accuracy
- Create a Streamlit app for predictions
- Use more recent climate datasets

---

## 👩‍💻 Author

**Jhansi Pydi**  
Full Stack Developer | AIML Student  
GitHub: [@Jhansi6135](https://github.com/Jhansi6135)

---

## 📜 License

This project is open source and free to use under the [MIT License](LICENSE).
