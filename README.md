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
  - pH → 491  
  - Sulfate → 781  
  - Trihalomethanes → 162  
- **Missing values after cleaning:** 0 in all columns  

---

## 🤖 Week 2 – Model Training & Evaluation

### 🔹 Models Implemented
- Logistic Regression
- Support Vector Machine (SVM, RBF kernel)
- K-Nearest Neighbors (k=7)
- Random Forest Classifier

### 🔹 Evaluation Metrics
- Used **Train/Test Split (80/20)** with Stratified Sampling
- Performed **Cross-Validation** (5-fold) for robustness
- Metrics considered: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Handled class imbalance with `class_weight="balanced"`

### 🔹 Results Summary
| Model                | CV Accuracy | Test Accuracy | Precision | Recall | F1   | ROC-AUC |
|-----------------------|-------------|---------------|-----------|--------|------|---------|
| Random Forest         | 0.674       | **0.6646**    | 0.6698    | 0.2773 | 0.3923 | **0.6576** |
| SVM (RBF)             | 0.6532      | 0.622         | 0.5159    | 0.5078 | 0.5118 | 0.6444 |
| KNN (k=7)             | 0.6404      | 0.5899        | 0.4570    | 0.2695 | 0.3391 | 0.5995 |
| Logistic Regression   | 0.5055      | 0.5259        | 0.4164    | 0.5352 | 0.4684 | 0.548  |

### 🔹 Key Findings
- **Random Forest** achieved the **highest Test Accuracy (66%)** and **ROC-AUC (65%)**.  
- Logistic Regression had slightly better recall but overall lower accuracy.  
- Random Forest was chosen as the **final model** for deployment.  

### 🔹 Outputs Generated
- **Best model saved:** `models/best_model.pkl`
- **Evaluation results:** `reports/results.csv`
- **Visualizations:**  
  - Confusion matrices for each model  
  - Accuracy comparison chart  
  - Feature importance plot (Random Forest)  

---

## 🚀 Week 3 – Deployment

### 🔹 Deployment Steps
1. Trained model was saved as `models/best_model.pkl` using **joblib**.  
2. A `predict.py` script was created to:  
   - Load the trained model  
   - Accept new input data (CSV file)  
   - Return predictions (Potable = 1 / Not Potable = 0)  
3. Example input file: `sample_input.csv` (without target column).  
4. Example output file: `predictions.csv` containing predictions.  

### 🔹 How to Run
```bash
# Install requirements
pip install -r requirements.txt

# Run predictions on new data
python predict.py sample_input.csv
