# Heart Disease Prediction — Artificial Neural Network (ANN)

**AI Assignment — Machine Learning (Supervised)**
**Method:** Artificial Neural Network (ANN)

---

## 1. Files Included


1. `train_ann_final.py` - Trains the ANN classifier on the heart disease dataset: loads and preprocesses the data, tunes hyperparameters, trains the final model, evaluates it, and saves the trained model for reuse.

2. `app.py` - Streamlit web interface that loads the trained model and lets a user enter a patient's information to receive a live heart disease risk prediction. 

3. `synthetic_heart_disease_dataset.csv` - The dataset used for training and evaluation . 

Running `train_ann_final.py` will also generate the following files, which `app.py` depends on:


 1. `heart_disease_ann_model.pkl` - The trained ANN model 
 2. `scaler.pkl` - Fitted `StandardScaler` used to scale input features 
 3. `feature_names.pkl` - Ordered list of feature column names expected by the model 
 4. `correlation_heatmap.png`- Feature correlation heatmap 
 5. `hyperparameter_tuning.png` - Cross-validated F1-score comparison across candidate ANN architectures 
 6. `ann_results.png` - Confusion matrix and ROC curve on the test set 

---

## 2. Requirements

- Python 3.11 (or compatible 3.x version)
- Packages:
  ```
  pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit
  ``` 

Tested with: numpy 2.4, pandas 3.0, scikit-learn 1.8, matplotlib 3.10, seaborn 0.13, streamlit 1.62, joblib 1.5. Other recent versions of these packages should also work.

---

## 3. How to Run

**Step 1 — Place the dataset**
Ensure `synthetic_heart_disease_dataset.csv` is in the same folder as `train_ann_final.py`.

**Step 2 — Train the model**
```
python train_ann_final.py
```
This will print dataset statistics, hyperparameter tuning results, and final test-set performance to the console, and will save the model files and result figures listed in Section 1.

**Step 3 — Launch the prediction interface**
Once training is complete (so that `heart_disease_ann_model.pkl`, `scaler.pkl`, and `feature_names.pkl` exist in the folder), run:
```
streamlit run app.py
```
(If `streamlit` is not recognised as a command, use `python -m streamlit run app.py` instead.)

This opens a browser window where a patient's demographic, lifestyle, medical history, and vitals/lab data can be entered to generate a live heart disease risk prediction with an associated probability.

