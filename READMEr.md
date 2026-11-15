# Campaign Revenue Prediction

A Machine Learning regression project that predicts the expected revenue of a marketing campaign based on planned campaign details using XGBoost Regression.

## Project files

- `campaign_revenue_prediction.ipynb` — notebook for data generation, inspection, preprocessing, feature engineering, model training, evaluation, and saving the model.
- `app.py` — Streamlit application for single-campaign and bulk CSV revenue predictions.
- `requirements.txt` — required Python packages.
- `data/` — generated raw campaign dataset (created when the notebook runs).
- `models/` — saved model artifact (created when the notebook runs).
- `outputs/` — evaluation metrics and test predictions (created when the notebook runs).

## Run the notebook

1. Install packages: `pip install -r requirements.txt`
2. Open `campaign_revenue_prediction.ipynb` in Jupyter.
3. Run all notebook cells in order.

The notebook generates a synthetic marketing campaign dataset for training and evaluation. The dataset contains campaign planning features such as planned budget, planned duration, planned audience size, offer discount percentage, campaign timing, and marketing channel.

The target variable is campaign revenue (`actual_revenue_inr`).

The target is generated for learning and demonstration purposes; it is not real business or customer data. High evaluation performance on synthetic data does not establish real-world business performance.

## Run the Streamlit app

After running the notebook:

```bash
streamlit run app.py
```

The application allows users to enter campaign planning details and estimate expected campaign revenue.

It also supports bulk predictions through CSV uploads.

The app uses the saved model artifact and its preprocessing components to ensure that input features are processed consistently with the training pipeline.

## Input features

The application accepts campaign planning features:

- `planned_budget_inr`
- `planned_duration_days`
- `planned_audience_size`
- `offer_discount_pct`
- `campaign_month`
- `campaign_year`
- `campaign_day_of_week`
- `channel`

The application does not require campaign ID or actual campaign revenue as prediction inputs.

## Input CSV columns

The uploaded CSV must contain the following columns:

- `planned_budget_inr`
- `planned_duration_days`
- `planned_audience_size`
- `offer_discount_pct`
- `campaign_month`
- `campaign_year`
- `campaign_day_of_week`
- `channel`

## Evaluation

The notebook evaluates model performance using regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Evaluation is performed using a separate holdout test dataset.

The model estimates campaign revenue in Indian Rupees (INR).

Performance metrics represent results on the generated synthetic dataset. They should not be interpreted as guaranteed revenue prediction accuracy in real-world marketing campaigns.

The test dataset is kept separate from model training and validation to provide an honest estimate of model performance.

## Technologies used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib