import io
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
SAMPLE_FILE = "SeoulBikeData_sample.csv"  # demo file committed next to app.py
MIN_ROWS = 20          # fewest complete rows we accept for a regression
MAX_MB = 50            # biggest upload we accept
MAX_PLOT_POINTS = 20000

st.set_page_config(page_title="CSV Predictor", layout="wide")
st.title("CSV Predictor")
st.write(
    "Upload any CSV file with at least three numeric columns. "
    "Explore it with a chart, then pick two inputs and a target and see what a straight-line "
    "(linear regression) model predicts."
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes):
    """Read a CSV from bytes. Returns (table, notes about anything we changed)."""
    notes = []
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        for sep in (None, ","):
            try:
                if sep is None:
                    candidate = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
                else:
                    candidate = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding)
            except Exception:  # noqa: BLE001 - we try the next option
                continue
            if candidate.shape[1] >= 2:
                df = candidate
                break
        if df is not None:
            break

    if df is None:
        raise ValueError(
            "Could not read this file as a table. Please upload a CSV with a header row "
            "and several columns."
        )

    # pandas renames repeated or blank headers silently; tell the user.
    repeated = [c for c in df.columns
                if re.match(r"^(.*)\.\d+$", str(c)) and re.match(r"^(.*)\.\d+$", str(c)).group(1) in list(df.columns)]
    blank = [c for c in df.columns if str(c).startswith("Unnamed:")]

    # Trim spaces, then make sure every column name is unique (two names that differ only
    # by spaces would otherwise collide and break the app).
    names, used = [], set()
    for original in df.columns:
        name = str(original).strip() or "Unnamed"
        new, k = name, 0
        while new in used:
            k += 1
            new = f"{name}.{k}"
        if new != name:
            repeated.append(new)
        used.add(new)
        names.append(new)
    df.columns = names

    if repeated:
        notes.append("Repeated column names were renamed by adding a number: " + ", ".join(sorted(set(map(str, repeated)))) + ".")
    if blank:
        notes.append(f"{len(blank)} column(s) had no header name and were called 'Unnamed'.")

    n_rows, n_cols = df.shape
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if df.shape[1] < n_cols:
        notes.append(f"Removed {n_cols - df.shape[1]} completely empty column(s).")
    if df.shape[0] < n_rows:
        notes.append(f"Removed {n_rows - df.shape[0]} completely empty row(s).")

    # Text columns that are really plain numbers (for example "1,234") become numeric.
    # Percentages and currency text are left alone, because their meaning is ambiguous.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.replace(r"[,\s]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            filled = int(df[col].notna().sum())
            if filled > 0 and int(converted.notna().sum()) >= 0.9 * filled:
                df[col] = converted
                notes.append(f"Column '{col}' was text that looks like numbers (such as 1,234), "
                             "so it was converted to numbers.")

    n_inf = int(np.isinf(df.select_dtypes(include="number")).sum().sum())
    if n_inf:
        notes.append(f"{n_inf} infinite value(s) were treated as missing.")
    return df.replace([np.inf, -np.inf], np.nan), notes


def as_dates(series: pd.Series):
    """Return the series as dates if (almost) all of it looks like a date, else None."""
    if pd.api.types.is_numeric_dtype(series):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if series.notna().sum() > 0 and parsed.notna().sum() >= 0.9 * series.notna().sum():
        return parsed
    return None


def default_index(options, preferred):
    """Index of the preferred column if this file has it, otherwise the first option."""
    return options.index(preferred) if preferred in options else 0


def shrink(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep charts fast on very large files by plotting every n-th row."""
    if len(frame) > MAX_PLOT_POINTS:
        step = math.ceil(len(frame) / MAX_PLOT_POINTS)
        st.caption(f"Large file: plotting every {step}th row to keep the chart fast.")
        return frame.iloc[::step]
    return frame


# ------------------------------------------------------------------
# Step 1. Load the data
# ------------------------------------------------------------------
st.header("1. Load your data")
uploaded_file = st.file_uploader(
    "Upload a CSV file", type=["csv", "txt"],
    help="The file needs a header row and at least three numeric columns.",
)

sample_available = os.path.exists(SAMPLE_FILE)
use_sample = False
if sample_available:
    left, right = st.columns([3, 1])
    use_sample = left.checkbox(
        "Use the built-in sample data (Seoul bike sharing, one row per hour) when no file is uploaded",
        value=True,
    )
    with open(SAMPLE_FILE, "rb") as f:
        right.download_button("Download the sample CSV", f.read(),
                              file_name=SAMPLE_FILE, mime="text/csv")

if uploaded_file is not None:
    raw = uploaded_file.getvalue()
    source_name = uploaded_file.name
elif sample_available and use_sample:
    with open(SAMPLE_FILE, "rb") as f:
        raw = f.read()
    source_name = SAMPLE_FILE
else:
    st.info("Upload a CSV file above to get started. It needs at least three numeric columns.")
    st.stop()

if len(raw) == 0:
    st.error("This file is empty. Please upload a CSV that contains data.")
    st.stop()
if len(raw) > MAX_MB * 1024 * 1024:
    st.error(f"This file is larger than {MAX_MB} MB. Please upload a smaller file.")
    st.stop()

try:
    df, cleaning_notes = read_csv_bytes(raw)
except ValueError as err:
    st.error(str(err))
    st.stop()

if df.empty:
    st.error("This file has a header but no data rows.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
usable_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
ignored_cols = [c for c in df.columns if c not in numeric_cols]

if len(numeric_cols) < 3:
    st.error(
        f"This file has only {len(numeric_cols)} numeric column(s). "
        "The app needs at least three numeric columns (two inputs and a target)."
        + (" Text columns that were not treated as numbers: " + ", ".join(ignored_cols) + "." if ignored_cols else "")
    )
    st.stop()
if len(usable_cols) < 3:
    st.error(
        "Fewer than three numeric columns have more than one distinct value, "
        "so there is nothing for a model to learn from."
    )
    st.stop()

st.success(
    f"Loaded {source_name}: {len(df):,} rows, {df.shape[1]} columns, "
    f"{len(usable_cols)} usable numeric columns."
)
if uploaded_file is not None and sample_available and use_sample:
    st.caption("Your uploaded file is being used instead of the sample data.")
if ignored_cols:
    st.caption("Text columns are skipped for charts and the model: " + ", ".join(ignored_cols) + ".")
if cleaning_notes:
    st.info("Automatic cleaning applied (the preview below shows the cleaned data):\n\n"
            + "\n".join(f"- {n}" for n in cleaning_notes))

st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), hide_index=True, width="stretch")

# ------------------------------------------------------------------
# Step 2. Explore a column
# ------------------------------------------------------------------
st.header("2. Explore a column")
chart_type = st.radio("Chart type", ["Line", "Scatter", "Histogram"], horizontal=True)

if chart_type == "Histogram":
    col = st.selectbox("Column", usable_cols, key="hist_col")
    bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins")
    fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
    st.plotly_chart(fig, width="stretch")

elif chart_type == "Scatter":
    a, b = st.columns(2)
    x_sc = a.selectbox("X axis", usable_cols, key="sc_x")
    y_sc = b.selectbox("Y axis", [c for c in usable_cols if c != x_sc], key="sc_y")
    pts = shrink(df[[x_sc, y_sc]].dropna())
    fig = px.scatter(pts, x=x_sc, y=y_sc, title=f"{y_sc} versus {x_sc}")
    st.plotly_chart(fig, width="stretch")

else:
    a, b = st.columns(2)
    y_ln = a.selectbox("Column to plot", usable_cols, key="ln_y")
    date_cols = {c: as_dates(df[c]) for c in ignored_cols}
    date_cols = {c: v for c, v in date_cols.items() if v is not None}
    hour_col = next((c for c in usable_cols if c.strip().lower() == "hour"
                     and df[c].dropna().between(0, 23).all()), None)
    combined = {f"{d} + {hour_col}": d for d in date_cols} if hour_col else {}
    x_options = list(combined) + ["Row number"] + list(date_cols) + usable_cols
    x_ln = b.selectbox("X axis", x_options, key="ln_x")

    if x_ln in combined:
        x_values = date_cols[combined[x_ln]] + pd.to_timedelta(df[hour_col], unit="h")
    elif x_ln == "Row number":
        x_values = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    elif x_ln in date_cols:
        x_values = date_cols[x_ln]
    else:
        x_values = df[x_ln]

    line_df = pd.DataFrame({"x": x_values, "y": df[y_ln]}).dropna()
    line_df = line_df.sort_values("x", kind="stable")
    if line_df["x"].duplicated().any():
        st.caption("The X axis has repeated values, so points with the same X are joined in file order.")
    line_df = shrink(line_df)
    fig = px.line(line_df, x="x", y="y", labels={"x": x_ln, "y": y_ln}, title=f"{y_ln} by {x_ln}")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------
# Step 3. Fit a model and predict
# ------------------------------------------------------------------
st.header("3. Predict with a straight line")
st.write("Pick two inputs (what you know) and one target (what you want to predict).")

c1, c2, c3 = st.columns(3)
input1 = c1.selectbox("Input 1", usable_cols,
                      index=default_index(usable_cols, "Hour"), key="in1")
options2 = [c for c in usable_cols if c != input1]
input2 = c2.selectbox("Input 2", options2,
                      index=default_index(options2, "Temperature(°C)"), key="in2")
options3 = [c for c in usable_cols if c not in (input1, input2)]
target = c3.selectbox("Target", options3,
                      index=default_index(options3, "Rented Bike Count"), key="target")

inputs = [input1, input2]
data = df[inputs + [target]].dropna()
dropped = len(df) - len(data)
if dropped:
    st.caption(f"{dropped:,} rows were skipped because one of these three columns is empty.")
if len(data) < MIN_ROWS:
    st.error(f"Need at least {MIN_ROWS} rows with values in all three columns, but found {len(data)}.")
    st.stop()
if any(data[c].nunique() < 2 for c in inputs + [target]):
    st.error("One of the chosen columns has a single repeated value in the rows used. Please pick another column.")
    st.stop()

# Warnings about the choice of columns
for c in inputs:
    corr = data[c].corr(data[target])
    if pd.notna(corr) and abs(corr) > 0.98:
        st.warning(
            f"{c} is almost identical to {target} (correlation {corr:.2f}). "
            "The fit will look excellent but may tell you very little."
        )
between = data[input1].corr(data[input2])
if pd.notna(between) and abs(between) > 0.9:
    st.warning(
        f"{input1} and {input2} move almost together (correlation {between:.2f}), "
        "so the model cannot separate their effects. Consider a different pair."
    )

# Fit. Column names and order are the same in training and in prediction.
X = data[inputs]
y = data[target]
model = LinearRegression().fit(X, y)

if len(data) >= 50:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    check_model = LinearRegression().fit(X_train, y_train)
    r2_eval = r2_score(y_test, check_model.predict(X_test))
    mae_eval = mean_absolute_error(y_test, check_model.predict(X_test))
    basis = "on 20% of rows the model did not see"
    eval_tag = "test rows"
else:
    r2_eval = model.score(X, y)
    mae_eval = mean_absolute_error(y, model.predict(X))
    basis = "on the same rows it learned from (small file)"
    eval_tag = "training rows"

avg_target = float(y.mean())
pct_error = (mae_eval / abs(avg_target) * 100) if avg_target != 0 else None

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"R² ({eval_tag})", f"{r2_eval:.2f}")
m2.metric(f"Typical error ({eval_tag})", f"{mae_eval:,.2f}")
m3.metric(f"Average {target}", f"{avg_target:,.2f}")
m4.metric("Rows used", f"{len(data):,}")
st.caption(f"R² and typical error are measured {basis}. "
           "Typical error is the average absolute miss; individual predictions can miss by more.")
if len(data) >= 50:
    st.caption("These numbers come from a model trained on 80% of the rows. "
               "The prediction below uses the same kind of model refitted on all usable rows.")
else:
    st.warning(f"Only {len(data)} usable rows: too few to hold back a separate test set, so the numbers above "
               "look better than the model will do on new data.")

if r2_eval < 0:
    st.warning(f"On the {eval_tag}, these inputs predicted {target} worse than simply using the average "
               "of those rows, so they are not useful for this target.")
elif r2_eval < 0.3:
    st.warning(f"The two inputs explain only about {r2_eval * 100:.0f}% of the variation in {target}, "
               "so predictions will often be far off.")
else:
    st.info(f"The two inputs explain about {r2_eval * 100:.0f}% of the variation in {target}. "
            "Whether that is good enough depends on how costly a wrong prediction is.")

with st.expander("How does this model work?"):
    st.write(
        f"Linear regression finds the straight-line formula that best matches the data: "
        f"**{target} = {model.intercept_:,.4g} + {model.coef_[0]:,.4g} × {input1} "
        f"+ {model.coef_[1]:,.4g} × {input2}**."
    )
    st.write(
        f"Holding {input2} fixed, each extra unit of {input1} changes the predicted {target} by "
        f"{model.coef_[0]:,.4g}. Holding {input1} fixed, each extra unit of {input2} changes it by "
        f"{model.coef_[1]:,.4g}. A straight line cannot capture curves or peaks in the data."
    )

actual_vs_pred = shrink(pd.DataFrame({"Predicted": model.predict(X), "Actual": y.values}))
fig = px.scatter(actual_vs_pred, x="Predicted", y="Actual",
                 labels={"Predicted": f"Predicted {target}", "Actual": f"Actual {target}"},
                 title="Actual versus predicted on all rows (in-sample; points on the line are perfect predictions)")
lo = float(min(actual_vs_pred["Predicted"].min(), actual_vs_pred["Actual"].min()))
hi = float(max(actual_vs_pred["Predicted"].max(), actual_vs_pred["Actual"].max()))
fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction")
st.plotly_chart(fig, width="stretch")

st.subheader("Try it: predict for your own values")
v1, v2 = st.columns(2)
val1 = v1.number_input(f"Value for {input1}", value=float(data[input1].mean()))
val2 = v2.number_input(f"Value for {input2}", value=float(data[input2].mean()))

new_row = pd.DataFrame({input1: [val1], input2: [val2]})[inputs]  # same names and order as training
prediction = float(model.predict(new_row)[0])

# Warnings come first so they are seen before the number.
for c, v in zip(inputs, (val1, val2)):
    low, high = float(data[c].min()), float(data[c].max())
    if v < low or v > high:
        st.warning(
            f"{c} = {v:,.2f} is outside the range the model learned from "
            f"({low:,.2f} to {high:,.2f}). Treat the prediction below with caution."
        )
if float(y.min()) >= 0 and prediction < 0:
    st.warning(f"The model predicts a negative {target}, but the data never goes below zero. "
               "A straight line can do this for unusual inputs.")

st.metric(f"Predicted {target}", f"{prediction:,.2f}")
st.caption(
    f"The model's average absolute error on {eval_tag} was {mae_eval:,.2f} {target} units. "
    "Individual predictions may have larger errors."
)

# ------------------------------------------------------------------
# The one sentence the assignment asks for
# ------------------------------------------------------------------
st.header("What is this model used for, and what if it is wrong?")
pct_text = f" ({pct_error:.0f}% of the average)" if pct_error is not None and pct_error <= 300 else ""
if target == "Rented Bike Count":
    consequence = ("if it under-predicts, stations run out of bikes at busy hours, and if it over-predicts, "
                   "bikes sit idle and money is wasted")
else:
    consequence = ("a decision based on it may be too high or too low, "
                   "especially for unusual or out-of-range values")
st.info(
    f"This model estimates **{target}** from **{input1}** and **{input2}** to give a rough planning "
    f"number, and when it is wrong its average miss is about {mae_eval:,.2f}{pct_text} "
    f"(individual misses can be larger), so {consequence}."
)
