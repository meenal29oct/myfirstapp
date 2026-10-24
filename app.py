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

# --------------------------------------------------
# Settings
# --------------------------------------------------
SAMPLE_FILE = "SeoulBikeData_sample.csv"
MIN_ROWS = 20
MAX_MB = 50
MAX_PLOT_POINTS = 20000

st.set_page_config(page_title="CSV Predictor", layout="wide")
st.title("CSV Predictor")
st.write(
    "Upload any CSV file with at least three numeric columns. "
    "Explore it with a chart, then pick two inputs and a target "
    "and see what a linear regression model predicts."
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes):
    notes = []
    df = None

    for encoding in ("utf-8-sig", "latin-1"):
        for sep in (None, ","):
            try:
                if sep is None:
                    candidate = pd.read_csv(
                        io.BytesIO(raw),
                        sep=None,
                        engine="python",
                        encoding=encoding
                    )
                else:
                    candidate = pd.read_csv(
                        io.BytesIO(raw),
                        sep=sep,
                        encoding=encoding
                    )
            except Exception:
                continue

            if candidate.shape[1] >= 2:
                df = candidate
                break

        if df is not None:
            break

    if df is None:
        raise ValueError(
            "Could not read this file as a table. "
            "Please upload a CSV with headers and several columns."
        )

    repeated = [
        c for c in df.columns
        if re.match(r"^(.*)\.\d+$", str(c))
        and re.match(r"^(.*)\.\d+$", str(c)).group(1)
        in list(df.columns)
    ]
    blank = [
        c for c in df.columns
        if str(c).startswith("Unnamed:")
    ]

    names = []
    used = set()

    for original in df.columns:
        name = str(original).strip() or "Unnamed"
        new = name
        k = 0

        while new in used:
            k += 1
            new = f"{name}.{k}"

        if new != name:
            repeated.append(new)

        used.add(new)
        names.append(new)

    df.columns = names

    if repeated:
        notes.append(
            "Repeated column names were renamed: "
            + ", ".join(sorted(set(map(str, repeated))))
            + "."
        )

    if blank:
        notes.append(
            f"{len(blank)} column(s) had no header name."
        )

    n_rows, n_cols = df.shape

    df = (
        df.dropna(axis=1, how="all")
        .dropna(axis=0, how="all")
        .reset_index(drop=True)
    )

    if df.shape[1] < n_cols:
        notes.append(
            f"Removed {n_cols - df.shape[1]} empty columns."
        )

    if df.shape[0] < n_rows:
        notes.append(
            f"Removed {n_rows - df.shape[0]} empty rows."
        )

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = (
                df[col].astype(str)
                .str.replace(r"[,\s]", "", regex=True)
            )
            converted = pd.to_numeric(
                cleaned, errors="coerce"
            )

            filled = int(df[col].notna().sum())

            if (
                filled > 0
                and int(converted.notna().sum()) >= 0.9 * filled
            ):
                df[col] = converted
                notes.append(
                    f"Column '{col}' was converted "
                    "from numeric-looking text to numbers."
                )

    n_inf = int(
        np.isinf(
            df.select_dtypes(include="number")
        ).sum().sum()
    )

    if n_inf:
        notes.append(
            f"{n_inf} infinite values were treated as missing."
        )

    return df.replace([np.inf, -np.inf], np.nan), notes


def as_dates(series):
    if pd.api.types.is_numeric_dtype(series):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True
        )

    nonmissing = series.notna().sum()

    if (
        nonmissing > 0
        and parsed.notna().sum() >= 0.9 * nonmissing
    ):
        return parsed

    return None


def default_index(options, preferred):
    return (
        options.index(preferred)
        if preferred in options else 0
    )


def shrink(frame):
    if len(frame) > MAX_PLOT_POINTS:
        step = math.ceil(
            len(frame) / MAX_PLOT_POINTS
        )
        st.caption(
            f"Large file: plotting every {step}th row."
        )
        return frame.iloc[::step]

    return frame


# --------------------------------------------------
# STEP 1 — Load data
# --------------------------------------------------
st.header("1. Load your data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv", "txt"],
    help="At least three numeric columns are required."
)

sample_available = os.path.exists(SAMPLE_FILE)
use_sample = False

if sample_available:
    left, right = st.columns([3, 1])

    use_sample = left.checkbox(
        "Use the built-in Seoul bike sharing sample "
        "when no file is uploaded",
        value=True
    )

    with open(SAMPLE_FILE, "rb") as f:
        right.download_button(
            "Download the sample CSV",
            f.read(),
            file_name=SAMPLE_FILE,
            mime="text/csv"
        )

if uploaded_file is not None:
    raw = uploaded_file.getvalue()
    source_name = uploaded_file.name

elif sample_available and use_sample:
    with open(SAMPLE_FILE, "rb") as f:
        raw = f.read()
    source_name = SAMPLE_FILE

else:
    st.info(
        "Upload a CSV with at least three numeric columns."
    )
    st.stop()

if len(raw) == 0:
    st.error("The uploaded file is empty.")
    st.stop()

if len(raw) > MAX_MB * 1024 * 1024:
    st.error(
        f"The file exceeds the {MAX_MB} MB limit."
    )
    st.stop()

try:
    df, cleaning_notes = read_csv_bytes(raw)
except ValueError as err:
    st.error(str(err))
    st.stop()

if df.empty:
    st.error(
        "The file has no usable data rows."
    )
    st.stop()

numeric_cols = (
    df.select_dtypes(include="number")
    .columns.tolist()
)

usable_cols = [
    c for c in numeric_cols
    if df[c].nunique(dropna=True) > 1
]

ignored_cols = [
    c for c in df.columns
    if c not in numeric_cols
]

if len(numeric_cols) < 3:
    st.error(
        f"Only {len(numeric_cols)} numeric columns found. "
        "At least three are required."
    )
    st.stop()

if len(usable_cols) < 3:
    st.error(
        "At least three numeric columns must contain "
        "more than one distinct value."
    )
    st.stop()

st.success(
    f"Loaded {source_name}: "
    f"{len(df):,} rows, "
    f"{df.shape[1]} columns, "
    f"{len(usable_cols)} usable numeric columns."
)

if uploaded_file is not None and sample_available and use_sample:
    st.caption(
        "Your uploaded file is being used instead of the sample."
    )

if ignored_cols:
    st.caption(
        "Text columns are excluded from charts and regression: "
        + ", ".join(ignored_cols)
    )

if cleaning_notes:
    st.info(
        "Automatic cleaning applied:\n\n"
        + "\n".join(f"- {n}" for n in cleaning_notes)
    )

st.subheader("Preview — first 10 rows")
st.dataframe(
    df.head(10),
    hide_index=True,
    width="stretch"
)


# --------------------------------------------------
# STEP 2 — Explore the data
# --------------------------------------------------
st.header("2. Explore a column")

chart_type = st.radio(
    "Chart type",
    ["Line", "Scatter", "Histogram"],
    horizontal=True
)

# HISTOGRAM — exact bin boundaries
if chart_type == "Histogram":
    col = st.selectbox(
        "Column",
        usable_cols,
        key="hist_col"
    )

    bins = st.slider(
        "Number of bins",
        5, 100, 30,
        key="hist_bins"
    )

    values = df[col].dropna().to_numpy()
    counts, edges = np.histogram(
        values,
        bins=bins
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    fig = px.bar(
        x=centers,
        y=counts,
        labels={
            "x": col,
            "y": "Number of observations"
        },
        title=f"Distribution of {col} ({bins} bins)"
    )

    fig.update_traces(
        width=np.diff(edges),
        marker_line_width=0
    )
    fig.update_layout(bargap=0)

    st.plotly_chart(
        fig,
        width="stretch"
    )

# SCATTER — prevent identical axes
elif chart_type == "Scatter":
    a, b = st.columns(2)

    x_sc = a.selectbox(
        "X axis",
        usable_cols,
        key="sc_x"
    )

    y_sc = b.selectbox(
        "Y axis",
        [
            c for c in usable_cols
            if c != x_sc
        ],
        key="sc_y"
    )

    pts = shrink(
        df[[x_sc, y_sc]].dropna()
    )

    fig = px.scatter(
        pts,
        x=x_sc,
        y=y_sc,
        opacity=0.4,
        title=f"{y_sc} versus {x_sc}"
    )

    fig.update_traces(
        marker=dict(size=5)
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# LINE — aggregate repeated X values
else:
    a, b = st.columns(2)

    y_ln = a.selectbox(
        "Column to plot",
        usable_cols,
        key="ln_y"
    )

    date_cols = {
        c: as_dates(df[c])
        for c in ignored_cols
    }

    date_cols = {
        c: v
        for c, v in date_cols.items()
        if v is not None
    }

    hour_col = next(
        (
            c for c in usable_cols
            if c.strip().lower() == "hour"
            and df[c].dropna().between(0, 23).all()
        ),
        None
    )

    combined = (
        {
            f"{d} + {hour_col}": d
            for d in date_cols
        }
        if hour_col else {}
    )

    x_options = (
        list(combined)
        + ["Row number"]
        + list(date_cols)
        + [
            c for c in usable_cols
            if c != y_ln
        ]
    )

    x_ln = b.selectbox(
        "X axis",
        x_options,
        key="ln_x"
    )

    if x_ln in combined:
        x_values = (
            date_cols[combined[x_ln]]
            + pd.to_timedelta(
                df[hour_col],
                unit="h"
            )
        )

    elif x_ln == "Row number":
        x_values = pd.Series(
            np.arange(1, len(df) + 1),
            index=df.index
        )

    elif x_ln in date_cols:
        x_values = date_cols[x_ln]

    else:
        x_values = df[x_ln]

    line_df = pd.DataFrame({
        "x": x_values,
        "y": df[y_ln]
    }).dropna()

    if line_df.empty:
        st.warning(
            "No valid observations for this chart."
        )

    else:
        chronological = (
            x_ln in combined
            or x_ln == "Row number"
        )

        if (
            not chronological
            and line_df["x"].duplicated().any()
        ):
            if (
                pd.api.types.is_numeric_dtype(
                    line_df["x"]
                )
                and line_df["x"].nunique() > 50
            ):
                line_df["group"] = pd.cut(
                    line_df["x"],
                    bins=50,
                    duplicates="drop"
                )

                line_df = (
                    line_df.groupby(
                        "group",
                        observed=True
                    )
                    .agg(
                        x=("x", "mean"),
                        y=("y", "mean")
                    )
                    .dropna()
                    .reset_index(drop=True)
                )

                st.caption(
                    "Showing averages across up to "
                    "50 X-axis ranges."
                )

            else:
                line_df = (
                    line_df.groupby(
                        "x",
                        as_index=False
                    )["y"].mean()
                )

                st.caption(
                    "Repeated X values are combined "
                    "using average Y values."
                )

        line_df = line_df.sort_values(
            "x",
            kind="stable"
        )

        line_df = shrink(line_df)

        fig = px.line(
            line_df,
            x="x",
            y="y",
            labels={
                "x": x_ln,
                "y": y_ln
            },
            title=f"{y_ln} by {x_ln}"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# --------------------------------------------------
# STEP 3 — Linear regression
# --------------------------------------------------
st.header("3. Predict with a straight line")

st.write(
    "Choose the target first, then two different "
    "input columns."
)

c1, c2, c3 = st.columns(3)

# Target is independent of input selections.
target = c1.selectbox(
    "Target",
    usable_cols,
    index=default_index(
        usable_cols,
        "Rented Bike Count"
    ),
    key="target"
)

options1 = [
    c for c in usable_cols
    if c != target
]

input1 = c2.selectbox(
    "Input 1",
    options1,
    index=default_index(
        options1,
        "Hour"
    ),
    key="in1"
)

options2 = [
    c for c in options1
    if c != input1
]

input2 = c3.selectbox(
    "Input 2",
    options2,
    index=default_index(
        options2,
        "Temperature(°C)"
    ),
    key="in2"
)

inputs = [input1, input2]

data = df[
    inputs + [target]
].dropna()

dropped = len(df) - len(data)

if dropped:
    st.caption(
        f"{dropped:,} rows skipped due to missing values."
    )

if len(data) < MIN_ROWS:
    st.error(
        f"At least {MIN_ROWS} complete rows required. "
        f"Only {len(data)} available."
    )
    st.stop()

if any(
    data[c].nunique() < 2
    for c in inputs + [target]
):
    st.error(
        "One selected column has only one distinct "
        "value in the complete rows."
    )
    st.stop()

# Check highly correlated columns
for c in inputs:
    corr = data[c].corr(data[target])

    if pd.notna(corr) and abs(corr) > 0.98:
        st.warning(
            f"{c} is highly correlated with {target} "
            f"(correlation {corr:.2f}). "
            "Interpret the model carefully."
        )

between = data[input1].corr(
    data[input2]
)

if pd.notna(between) and abs(between) > 0.9:
    st.warning(
        f"{input1} and {input2} are highly correlated "
        f"({between:.2f}). Their individual coefficients "
        "may be difficult to interpret."
    )

# Training and prediction use identical column order
X = data[inputs]
y = data[target]

model = LinearRegression()
model.fit(X, y)

# Hold-out evaluation for larger datasets
if len(data) >= 50:
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    check_model = LinearRegression()
    check_model.fit(
        X_train,
        y_train
    )

    test_predictions = check_model.predict(
        X_test
    )

    r2_eval = r2_score(
        y_test,
        test_predictions
    )

    mae_eval = mean_absolute_error(
        y_test,
        test_predictions
    )

    basis = (
        "on the 20% of rows not used "
        "to train the evaluation model"
    )
    eval_tag = "test rows"

else:
    fitted = model.predict(X)

    r2_eval = r2_score(y, fitted)
    mae_eval = mean_absolute_error(
        y,
        fitted
    )

    basis = "on the training rows"
    eval_tag = "training rows"

avg_target = float(y.mean())

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    f"R² ({eval_tag})",
    f"{r2_eval:.2f}"
)

m2.metric(
    f"Typical error ({eval_tag})",
    f"{mae_eval:,.2f}"
)

m3.metric(
    f"Average {target}",
    f"{avg_target:,.2f}"
)

m4.metric(
    "Rows used",
    f"{len(data):,}"
)

st.caption(
    f"R² and typical error are measured {basis}. "
    "Typical error is mean absolute error (MAE). "
    "Individual errors can be larger."
)

if len(data) >= 50:
    st.caption(
        "The metrics use an 80/20 train/test split. "
        "The interactive prediction below uses a "
        "model refitted on all complete rows."
    )
else:
    st.warning(
        "This dataset is too small for a separate "
        "test set, so the training metrics may "
        "overstate performance on new data."
    )

if r2_eval < 0:
    st.warning(
        "This model performed worse than predicting "
        "the test-set average. It may not generalize "
        "reliably to new data."
    )

elif r2_eval < 0.3:
    st.warning(
        f"The model explains only about "
        f"{r2_eval * 100:.0f}% of the variation "
        f"in {target}. Predictions may be inaccurate."
    )

else:
    st.info(
        f"The model explains approximately "
        f"{r2_eval * 100:.0f}% of the variation "
        f"in {target}. Whether this is sufficient "
        "depends on the consequences of errors."
    )

# Explain the fitted model
with st.expander("How does this model work?"):

    def signed_term(coefficient, name):
        sign = "+" if coefficient >= 0 else "−"
        return (
            f" {sign} "
            f"{abs(coefficient):,.4g} × {name}"
        )

    formula = (
        f"{target} = {model.intercept_:,.4g}"
        f"{signed_term(model.coef_[0], input1)}"
        f"{signed_term(model.coef_[1], input2)}"
    )

    st.write(
        "Linear regression finds a straight-line "
        "formula that approximates the relationship "
        "between the inputs and target."
    )

    st.code(
        formula,
        language=None
    )

    st.write(
        f"Holding {input2} fixed, one additional "
        f"unit of {input1} changes the predicted "
        f"{target} by {model.coef_[0]:,.4g}."
    )

    st.write(
        f"Holding {input1} fixed, one additional "
        f"unit of {input2} changes the predicted "
        f"{target} by {model.coef_[1]:,.4g}."
    )

    st.caption(
        "These coefficients describe associations, "
        "not necessarily causal effects."
    )

# Actual vs predicted chart
actual_vs_pred = pd.DataFrame({
    "Predicted": model.predict(X),
    "Actual": y.to_numpy()
})

actual_vs_pred = shrink(
    actual_vs_pred
)

fig = px.scatter(
    actual_vs_pred,
    x="Predicted",
    y="Actual",
    opacity=0.4,
    labels={
        "Predicted": f"Predicted {target}",
        "Actual": f"Actual {target}"
    },
    title=(
        "Actual versus predicted — in-sample "
        "(points on diagonal are perfect predictions)"
    )
)

lo = float(
    min(
        actual_vs_pred["Predicted"].min(),
        actual_vs_pred["Actual"].min()
    )
)

hi = float(
    max(
        actual_vs_pred["Predicted"].max(),
        actual_vs_pred["Actual"].max()
    )
)

fig.add_scatter(
    x=[lo, hi],
    y=[lo, hi],
    mode="lines",
    name="Perfect prediction"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# Interactive predictions
st.subheader(
    "Try it: predict for your own values"
)

v1, v2 = st.columns(2)


def sensible_default(column):
    median = float(data[column].median())
    values = data[column].dropna().to_numpy()

    if np.all(
        np.isclose(
            values,
            np.round(values)
        )
    ):
        return float(round(median))

    return float(round(median, 2))


val1 = v1.number_input(
    f"Value for {input1}",
    value=sensible_default(input1),
    key=f"val1_{input1}"
)

val2 = v2.number_input(
    f"Value for {input2}",
    value=sensible_default(input2),
    key=f"val2_{input2}"
)

# Preserve the training feature names and order
new_row = pd.DataFrame({
    input1: [val1],
    input2: [val2]
})[inputs]

prediction = float(
    model.predict(new_row)[0]
)

# Warn about extrapolation
for c, v in zip(
    inputs,
    (val1, val2)
):
    low = float(data[c].min())
    high = float(data[c].max())

    if v < low or v > high:
        st.warning(
            f"{c} = {v:,.2f} is outside the "
            f"observed range "
            f"({low:,.2f} to {high:,.2f}). "
            "Treat this prediction with caution."
        )

if (
    float(y.min()) >= 0
    and prediction < 0
):
    st.warning(
        f"The model predicts a negative {target}, "
        "although the observed values are nonnegative. "
        "Linear regression can produce unrealistic "
        "predictions for some inputs."
    )

st.metric(
    f"Predicted {target}",
    f"{prediction:,.2f}"
)

st.caption(
    f"Average absolute error on {eval_tag}: "
    f"{mae_eval:,.2f} {target} units. "
    "Individual errors may be larger."
)

# --------------------------------------------------
# REQUIRED BUSINESS PURPOSE / ERROR SENTENCE
# --------------------------------------------------
st.header(
    "What is this model used for, "
    "and what if it is wrong?"
)

if target == "Rented Bike Count":
    purpose = (
        f"This model estimates hourly bike rentals "
        f"using {input1} and {input2} to support "
        "bike availability planning; underprediction "
        "can cause shortages and missed rentals, "
        "while overprediction can leave bikes idle "
        "and increase operating costs."
    )

else:
    purpose = (
        f"This model estimates {target} using "
        f"{input1} and {input2} for exploratory "
        "planning; inaccurate predictions may "
        "lead to poor decisions if the estimates "
        "are used without validation."
    )

st.info(purpose)

st.caption(
    f"Model evaluation MAE ({eval_tag}): "
    f"{mae_eval:,.2f} {target} units."
)
