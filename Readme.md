# Restaurant Rating Prediction using Machine Learning

A machine learning system that predicts restaurant ratings **and** explains *why* — giving actionable insights to restaurant owners and operators.

---

## What Makes This Different

Most restaurant-rating projects stop at "predict a number." This project goes further:

- **SHAP Explainability** — understand *which features* drive each prediction
- **What-If Scenarios** — simulate adding table booking or delivery and see the rating impact
- **Market Benchmarking** — compare your restaurant against city-level averages
- **Business Insights Engine** — auto-generated recommendations based on your inputs
- **Model Comparison Dashboard** — visual comparison of 6 ML algorithms

---

## ML Pipeline

| Stage | Detail |
|---|---|
| Algorithms | Linear Regression, SVR, Decision Tree, Random Forest, KNN, AdaBoost |
| Tuning | GridSearchCV with cross-validation |
| Best Model | Random Forest Regressor |
| Explainability | SHAP TreeExplainer |
| Metrics | MAE, RMSE |

---

## Tech Stack
- Python  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- Scikit-learn
- Streamlit
