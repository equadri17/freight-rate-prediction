# Freight Rate Prediction

Machine learning solution for predicting freight rates from historical load data.

## Problem

The objective is to predict the freight rate for unseen loads using historical labeled development data.

The assessment provides:

- `train_test.csv` — labeled development data
- `validation.csv` — 12,000 loads requiring predictions
- `validation_predictions_template.csv` — required prediction template
- `december_chart_inputs.csv` — fixed December prediction scenario
- `score.py` — submission validation script

## Approach

The solution follows the following pipeline:

1. Exploratory data analysis
2. Data quality checks
3. Feature engineering
4. Time-based train/validation split
5. Regression model comparison
6. Model selection
7. Final model training on the complete development dataset
8. Prediction of the 12,000 validation loads
9. December predictions
10. Submission validation using the provided scorer

## Feature Engineering

The model uses the original numerical and categorical variables along with engineered features including:

- Date components
- Day of week
- Day of year
- Cyclical date features
- Log-transformed distance
- Squared distance
- Log-transformed weight
- Weight-to-distance ratio
- Pickup-delivery route

Categorical variables such as pickup, delivery, equipment and route are handled by the selected model.

## Validation Strategy

A chronological validation approach was used to better simulate prediction of future freight loads.

The development data was divided into:

- Training: historical data before the validation period
- Validation: a later chronological period

This avoids randomly mixing future observations into the training data.

## Model

Several regression approaches were considered during experimentation.

The final model was selected based on validation performance.

The final training pipeline uses CatBoost regression with categorical features and a log-transformed target.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
