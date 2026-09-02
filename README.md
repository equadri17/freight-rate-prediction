Freight Rate Prediction

Machine Learning system for predicting freight/shipping rates using historical load, route, distance, equipment, weight, market, and quote-related features.

 Project Overview

Freight pricing varies significantly depending on factors such as transportation distance, pickup and delivery locations, equipment type, shipment weight, market conditions, and quote signals.

This project develops an end-to-end freight rate prediction pipeline that learns pricing patterns from historical shipment data and predicts the expected posted freight rate for new loads.

The project covers:

Data exploration and quality analysis
Feature engineering
Baseline and machine learning model comparison
Chronological validation
CatBoost regression
Log-transformed target modeling
Validation prediction generation
December daily rate forecasting
Model evaluation and output validation

 Objective

The main objective is to predict:

posted_rate

for a freight load based on information available about the shipment.

Input Features

The dataset contains information such as:

Feature	Description
pickup	Pickup location
delivery	Delivery location
pickup_lat	Pickup latitude
pickup_lon	Pickup longitude
delivery_lat	Delivery latitude
delivery_lon	Delivery longitude
distance	Shipment distance
equipment	Equipment type
weight	Shipment weight
date	Load date
market_index	Market condition indicator
quote_signal	Quote-related signal
Target
posted_rate

 Exploratory Data Analysis

The analysis investigated:

Dataset dimensions and column types
Missing values
Duplicate records
Numerical feature distributions
Categorical variables
Target distribution
Correlations between numerical variables
Relationship between distance and freight rate
Time-based patterns

One of the strongest relationships observed was between distance and posted rate, with a correlation of approximately:

0.9085

This indicates that shipment distance is an important predictor of freight pricing.

 Feature Engineering

Several additional features were created to help the model capture nonlinear relationships and temporal patterns.

Date Features

From the original date column:

year
month
day
dayofweek
dayofyear
weekofyear

Cyclical features were also created:

sin_doy
cos_doy
sin_dow
cos_dow

These allow the model to represent recurring seasonal and weekly patterns.

Distance Features
distance_log
distance_sq
Weight Features
weight_log
Interaction Feature
weight_distance_ratio
Route Feature

A route identifier was created from pickup and delivery locations:

pickup_delivery

This allows the model to learn route-specific pricing patterns.

  Models Evaluated

Multiple approaches were compared using a chronological holdout set.

Model	MAE	RMSE	R²
Median Baseline	1146.79	1567.97	-0.052
Ridge Regression	410.82	782.83	0.738
Random Forest	188.51	704.23	0.788
CatBoost — Raw Target	140.00	649.57	0.819
CatBoost — Log Target	108.68	646.50	0.821

The metrics above come from the development/chronological holdout evaluation and are separate from the final unlabeled validation prediction set.

  Final Model

The final model uses:

CatBoostRegressor

with the following configuration:

Iterations:      300
Depth:            6
Learning Rate:    0.05
L2 Regularization: 5
Loss Function:    RMSE
Random Seed:      42

The target was transformed using:

np.log1p(posted_rate)

Predictions were converted back to the original rate scale using:

np.expm1(prediction)

This log-target approach produced substantially better MAE than the raw-target version during development validation.

  Validation Strategy

Because freight rates can change over time, a random train/test split can introduce unrealistic information leakage.

Instead, the project uses chronological validation.

The development data was divided using:

Training data:  Before October 2025
Holdout data:   October 2025 onward

This provides a more realistic approximation of the problem:

Train on historical loads → predict future loads.

The final assessment validation dataset contains 12,000 loads for which the target rate is not provided.

 December Forecast

The project also generates daily freight-rate predictions for a fixed shipment scenario across December.

The scenario contains:

Pickup:       Lexington
Delivery:     Fort Wayne
Distance:     360
Equipment:    Dry Van
Weight:       32,000

Predictions are generated for:

December 1 → December 31

The resulting visualization is available here:

  Project Structure
freight-rate-prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   ├── december_chart_inputs.csv
│   └── README.md
│
├── notebooks/
│   └── freight_rate_analysis.ipynb
│
├── src/
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   └── README.md
│
├── outputs/
│   ├── validation_predictions.csv
│   ├── december_predictions.csv
│   └── candidate_december.png
│
└── reports/
    └── Freight_Rate_Prediction_Report.pdf

 Getting Started
1. Clone the repository
git clone https://github.com/equadri17/freight-rate-prediction.git
cd freight-rate-prediction
2. Create a virtual environment
python -m venv venv

Activate it:

Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Train the model
python src/train.py

The pipeline performs feature engineering, trains the CatBoost model, generates predictions, and validates the resulting outputs.

  Results

The model achieved the following development holdout performance:

MAE  : 108.68
RMSE : 646.50
R²   : 0.821

Compared with the median baseline:

Baseline MAE : 1146.79
Model MAE    : 108.68

This demonstrates a substantial improvement over a simple baseline and shows that shipment and market-related features contain strong predictive information for freight pricing.

 Data & Reproducibility

The project uses structured freight-load data containing shipment, geographic, equipment, market, and pricing information.

Some datasets and assessment materials may be subject to their original usage or distribution restrictions. Please verify that you have permission to publicly redistribute any data or assessment materials before publishing them.

For a public portfolio repository, sensitive or restricted data should be removed or replaced with an appropriate sample dataset.

 Future Improvements

Potential improvements include:

Hyperparameter optimization
More extensive time-series validation
Route-level historical pricing features
Geographic distance engineering
Prediction uncertainty estimation
Model monitoring
FastAPI prediction service
Docker containerization
Interactive web dashboard
Cloud deployment
Automated CI/CD pipeline
 
  Detailed Report

A detailed project report covering the methodology, analysis, feature engineering, model experiments, evaluation, and results is available in:

reports/Freight_Rate_Prediction_Report.pdf
 
  Author

Sayyad Emad Husain Quadri

B.Tech — Information Technology

Interested in:

Machine Learning
Artificial Intelligence
Data Science
Cloud Computing
Software Engineering

The fixed December chart is:

scorer_results/candidate_december.png
Author

Sayyad Emad Husain Quadri
