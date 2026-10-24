# CSV Predictor

**Live app:** https://myfirstapp-1.streamlit.app/

**GitHub repository:** https://github.com/meenal29oct/myfirstapp

A Streamlit application for exploring CSV data and predicting a numeric target using linear regression.

## What it does

1. Upload a CSV file with at least three numeric columns and preview the data.
2. Explore numeric columns using line charts, scatter plots, or histograms.
3. Select one target and two input columns, fit a linear regression model, and enter custom values to generate predictions.
4. View model performance using R² and mean absolute error (MAE), along with warnings about unusual inputs and prediction limitations.

## Sample data

The application includes `SeoulBikeData_sample.csv`, based on the publicly available **Seoul Bike Sharing Demand** dataset from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

The sample is loaded automatically when no other CSV is uploaded.

**Default prediction setup:**
- Target: Rented Bike Count
- Input 1: Hour
- Input 2: Temperature(°C)

## Model evaluation

The application uses linear regression with two selected predictors.

For datasets containing at least 50 complete observations, model performance is evaluated using an 80/20 train/test split. The interactive prediction model is then fitted using all complete observations.

For smaller datasets, evaluation metrics are calculated on the training data and should be interpreted cautiously.

## Limitations

- A two-input linear regression model may not capture complex relationships, including rush-hour demand patterns.
- The sample model explains only part of the variation in bike rentals.
- Predictions outside the observed input ranges may be unreliable.
- Linear regression can produce negative predictions even when the target represents nonnegative quantities.
- Model coefficients represent associations, not necessarily causal relationships.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files

- `app.py` — Streamlit application and prediction logic.
- `requirements.txt` — Python dependencies.
- `SeoulBikeData_sample.csv` — Sample dataset.
- `README.md` — Project documentation.
