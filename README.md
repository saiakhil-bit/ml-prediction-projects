# Lead Conversion Prediction

A Logistic Regression project that estimates whether a lead converts within 30 days.

## Project files

- `lead_conversion_prediction.ipynb` — exactly 20 code cells for data generation, inspection, preprocessing, training, evaluation, and saving the model.
- `app.py` — Streamlit application for single-lead and bulk CSV predictions.
- `requirements.txt` — required Python packages.
- `data/` — generated raw dataset (created when the notebook runs).
- `models/` — saved model artifact (created when the notebook runs).
- `outputs/` — evaluation metrics and test predictions (created when the notebook runs).

## Run the notebook

1. Install packages: `pip install -r requirements.txt`
2. Open `lead_conversion_prediction.ipynb` in Jupyter.
3. Run all 20 cells in order.

The notebook creates a synthetic dataset of 50,000 leads. The target is generated for learning and demonstration; it is not real customer data. High accuracy on synthetic data does not establish real-world business performance.

## Run the Streamlit app

After running the notebook:

```bash
streamlit run app.py
```

The app accepts only the four model input features. It does not ask for lead ID, lead creation date, or campaign ID.

## Input CSV columns

- `prior_website_visits`
- `prior_email_engagement_count`
- `channel`
- `product_interest`

## Evaluation

The notebook reports accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, PR-AUC, and a calibration curve. It selects a follow-up threshold using validation data, based on an example capacity of 20% of leads.

Accuracy is measured honestly on the chronological holdout test set. A 95%+ score cannot be guaranteed; do not change the test labels or use future information to force a target score.
