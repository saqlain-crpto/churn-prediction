# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, with a Streamlit web app for live predictions.

## Results
- Models compared with 5-fold cross-validation: Logistic Regression (AUC 0.846), Random Forest (0.822), XGBoost (0.847)
- Test ROC-AUC: 0.843
- Churn rate in the data: 26.5% (imbalanced), so ROC-AUC and recall were used instead of accuracy
- At threshold 0.5: churn precision 0.53, recall 0.79, F1 0.63
- Threshold tuning: 0.3 gives recall 0.90 (precision 0.44); 0.7 gives precision 0.63 (recall 0.56). The best choice depends on retention-offer cost vs customer value.

## Key findings
- Month-to-month contracts are the strongest churn driver
- Fiber optic customers and customers without online security or tech support churn more
- The decision threshold is a business decision: a cheap retention offer favors a lower threshold (catch more churners), an expensive one favors a higher threshold (fewer false alarms)

## Project structure
- `data/` Telco Customer Churn dataset from Kaggle (not included in the repo)
- `notebooks/01_explore.ipynb` data exploration
- `src/train.py` trains and compares models, saves the best pipeline
- `src/evaluate.py` threshold tuning and feature importance
- `src/app.py` Streamlit app

## How to run
1. Download the dataset from Kaggle (Telco Customer Churn) and put the CSV in `data/`
2. Create a virtual environment and run `pip install -r requirements.txt`
3. `python src/train.py`
4. `streamlit run src/app.py`

## Limitations
- Feature importance shows what the model uses, not what causes churn
- In the app, TotalCharges is approximated as tenure x monthly charges