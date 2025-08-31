# 💧 Water Quality Classification (ML Project)

## 📌 Domain
Environment Monitoring and Pollution Control

## 📌 Objective
Predict whether water is **potable (safe to drink)** or not based on chemical parameters using Machine Learning.

---

## 📂 Dataset
Source: [Kaggle - Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)  

- **Rows (before cleaning):** 3276  
- **Columns:** 10  
- **Features:**  
  - pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity  
- **Target:**  
  - Potability → `0 = Not Drinkable`, `1 = Drinkable`

---

## ⚙️ Preprocessing
- Checked for missing values → Found in: `pH`, `Sulfate`, `Trihalomethanes`
- Filled missing values with **column mean**
- Normalized features using **StandardScaler**
- Saved cleaned dataset as: `water_potability_preprocessed.csv`

### 🔹 Outputs from Preprocessing
- **Original shape:** (3276, 10)  
- **Shape after cleaning:** (3276, 10)  
- **Missing values before cleaning:**
pH 491
Sulfate 781
Trihalomethanes 162
- **Missing values after cleaning:**  
All columns: 0
