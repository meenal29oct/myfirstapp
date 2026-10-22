# CSV Predictor

CSV Predictor is a Streamlit application for exploring CSV data and making predictions using linear regression.

## Features

- Upload and preview CSV files.
- Explore numeric columns using line, scatter, and histogram charts.
- Select two numeric input columns and one target column.
- Fit a linear regression model.
- Evaluate model performance using R² and mean absolute error (MAE).
- Enter custom input values to generate predictions.
- View warnings for unusual inputs and limitations of the model.

## Sample dataset

Seoul Bike Sharing Demand from the UCI Machine Learning Repository.

The default model predicts Rented Bike Count using Hour and Temperature(°C).

This is an introductory predictive model, not a production-grade demand forecasting system.

## Other datasets

The app also accepts other CSV files containing at least three usable numeric columns.

## Run locally

pip install -r requirements.txt

streamlit run app.py
