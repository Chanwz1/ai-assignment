# Combined Heart Disease Prediction App

This folder combines the ANN, KNN, and SVM prediction workflows into one Streamlit menu UI.

## Run

```bash
cd Heart_Disease_Combined
python -m streamlit run app.py
```

Use the sidebar to open the prediction page and choose `ANN`, `KNN`, `SVM`, or `Compare all`.

## Notes

- The app first looks for dataset and model files inside this folder.
- If a file is not copied here, it falls back to the original `ANN`, `KNN`, and `SVM` folders.
- KNN did not have a saved model artifact, so the combined app trains and saves `models/knn_pipeline.joblib` the first time KNN is used.
- Run `python copy_assets.py` if you retrain ANN or SVM and want to refresh the copied artifacts in this folder.
- This is an academic synthetic-data demo only, not a medical diagnostic tool.
