# Freight Rate Prediction

Machine learning solution for the Freight Rate Prediction Challenge.

The objective is to predict the freight rate (`posted_rate`) for unseen freight loads using the provided labeled development data.

## Project Overview

This project covers:

- Exploratory Data Analysis (EDA)
- Data-quality analysis
- Feature engineering
- Chronological train/validation splitting
- Regression model comparison
- CatBoost target transformation
- Hyperparameter tuning
- Final model training
- Prediction of 12,000 validation loads
- Prediction of the fixed December scenario
- Validation using the provided `score.py`

## Dataset

The assessment provides:

- `train_test.csv` — labeled development data
- `validation.csv` — 12,000 loads requiring predictions
- `validation_predictions_template.csv` — prediction template
- `december_chart_inputs.csv` — fixed December prediction scenario

The assessment data is not included in this repository.

Place the provided files in:

```text
data/
├── train_test.csv
├── validation.csv
├── validation_predictions_template.csv
└── december_chart_inputs.csv
Data

The development dataset contains the following fields:

load_id
pickup
delivery
pickup_lat
pickup_lon
delivery_lat
delivery_lon
distance
equipment
weight
date
market_index
quote_signal
posted_rate

posted_rate is the prediction target.

Data Quality

Missing numerical values were found in:

weight
market_index

The selected CatBoost model can handle missing numerical values natively, so these observations were retained rather than removed.

The categorical fields:

pickup
delivery
equipment

did not contain missing values in the inspected training and validation data.

Validation Strategy

A chronological validation strategy was used.

The development data was ordered by date and divided into an earlier training period and a later validation period.

This approach was chosen to better represent the real prediction setting, where the model is trained on historical loads and used to predict future loads.

Feature Engineering

The modeling pipeline uses the original numerical and categorical load attributes together with engineered features derived from the date and route information.

Feature engineering includes:

Date components
Day-of-week information
Day-of-year information
Cyclical date features
Distance transformations
Weight transformations
Weight-to-distance relationships
Pickup-delivery route information

The same feature-engineering logic is applied to the development, validation and December prediction data.

Model Experiments

Several models were evaluated using the same chronological validation approach.

Model	MAE	RMSE	R²
Ridge Regression	410.82	782.83	0.7377
Random Forest	188.51	704.23	0.7877
CatBoost — Raw Target	140.00	649.57	0.8194
CatBoost — Log Target	108.68	646.50	0.8211

CatBoost with a log-transformed target produced substantially better validation performance than the linear and Random Forest baselines.

Final Model

The final model uses CatBoostRegressor with the target transformed using:

np.log1p(posted_rate)

The selected hyperparameters are:

iterations     = 300
depth          = 6
learning_rate  = 0.05
l2_leaf_reg    = 5
random_seed    = 42

The log predictions are converted back to the original rate scale using:

np.expm1(prediction)
Final Validation Performance

Using the chronological validation set, the selected configuration achieved:

MAE  = 107.70
RMSE = 648.20
R²   = 0.8202

The model was then retrained using the complete labeled development dataset before generating the final 12,000 validation predictions.

Installation

Python 3.10+ is recommended.

Install the project dependencies:

python -m pip install -r requirements.txt
Requirements

The required Python packages are listed in:

requirements.txt

The main dependencies are:

pandas
numpy
matplotlib
scikit-learn
catboost
Running the Model

Place the assessment data in the data/ directory and run:

python src/train.py

The script generates:

validation_predictions.csv
december_chart_inputs_completed.csv
Validating the Predictions

Run the provided assessment scorer:

python score.py \
    --predictions validation_predictions.csv \
    --december-predictions december_chart_inputs_completed.csv

The scorer validates:

12,000 validation predictions
Required validation load_id values
Positive prediction values
31 December predictions
December dates from December 1 through December 31, 2025
Fixed December scenario inputs

The scorer creates:

scorer_results/candidate_december.png

The provided scorer confirms that the final validation metrics are calculated by Spotter after submission.

Repository Structure
freight-rate-prediction/
│
├── README.md
├── requirements.txt
├── score.py
│
├── src/
│   └── train.py
│
├── notebooks/
│   └── freight_rate_analysis.ipynb
│
└── scorer_results/
    └── candidate_december.png
Reproducibility

The notebook contains the exploratory analysis, model experiments and validation results.

The src/train.py script contains the final reproducible training and prediction pipeline.

Assessment Outputs

The final submission includes:

validation_predictions.csv

with exactly:

load_id,predicted_rate

The fixed December chart is:

scorer_results/candidate_december.png
Author

Sayyad Emad Husain Quadri
