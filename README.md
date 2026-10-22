# CSV Predictor

A Streamlit app that lets you upload a CSV file, explore a column with a line, scatter or histogram chart,
and fit a linear regression with two inputs and a target to predict values you type in.

- Sample data: Seoul Bike Sharing Demand (UCI Machine Learning Repository), one row per hour.
- Default model: Hour and Temperature predict Rented Bike Count. It explains roughly 40% of the variation,
  so it is a rough planning guide, not a precise forecast.
- Works with other CSV files that have at least three numeric columns.
- Run locally: `pip install -r requirements.txt` then `streamlit run app.py`
