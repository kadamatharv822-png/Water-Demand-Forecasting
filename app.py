
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Water Demand Forecasting",
    page_icon="💧",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💧 Water Demand Forecasting")

st.subheader(
    "Shripalavan, Taluka Man, District Satara, Maharashtra"
)

st.write(
    "AI & Machine Learning based Water Demand Forecasting System"
)

st.success("Water Demand Forecasting Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "water_demand.csv"
MODEL_PATH = "water_demand_model.pkl"

try:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Error loading data/model: {e}")
    st.stop()

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.header("📍 Project Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Population", "520")

with col2:
    st.metric("Households", "125")

with col3:
    st.metric(
        "Average Daily Supply",
        "25,000 L"
    )

with col4:
    st.metric(
        "Water / Household",
        "400 L"
    )

st.info(
    "Current supply pattern: Alternate-day water supply. "
    "Approximately 63 households receive 25,200 L and "
    "62 households receive 24,800 L."
)

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.header("📊 Dataset Preview")

st.write(
    f"Dataset contains {len(df)} daily records "
    f"from {df['Date'].min().date()} to {df['Date'].max().date()}."
)

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# DAILY WATER SUPPLY GRAPH
# --------------------------------------------------

st.header("📈 Daily Water Supply")

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    df["Date"],
    df["Total_Water_Supplied_L"],
    marker="o"
)

ax1.set_title("Daily Water Supply - Shripalavan")
ax1.set_xlabel("Date")
ax1.set_ylabel("Water Supplied (Litres)")
ax1.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig1)

# --------------------------------------------------
# WEATHER VS WATER SUPPLY
# --------------------------------------------------

st.header("🌦️ Weather Analysis")

col1, col2 = st.columns(2)

with col1:

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.scatter(
        df["Temperature"],
        df["Total_Water_Supplied_L"]
    )

    ax2.set_title("Temperature vs Water Supply")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Water Supply (Litres)")
    ax2.grid(True)

    st.pyplot(fig2)

with col2:

    fig3, ax3 = plt.subplots(figsize=(7, 5))

    ax3.scatter(
        df["Rainfall"],
        df["Total_Water_Supplied_L"]
    )

    ax3.set_title("Rainfall vs Water Supply")
    ax3.set_xlabel("Rainfall (mm)")
    ax3.set_ylabel("Water Supply (Litres)")
    ax3.grid(True)

    st.pyplot(fig3)

# --------------------------------------------------
# 7 DAY FUTURE FORECAST
# --------------------------------------------------

st.header("🔮 7-Day Water Demand Forecast")

future_dates = pd.date_range(
    start=df["Date"].max() + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Households_Supplied": [
        63 if i % 2 == 0 else 62
        for i in range(7)
    ],
    "Water_Per_Household_L": [400] * 7
})

future_df["Day"] = future_df["Date"].dt.day
future_df["Month"] = future_df["Date"].dt.month
future_df["Day_of_Week"] = future_df["Date"].dt.dayofweek

# Using last available weather values
future_df["Temperature"] = df["Temperature"].iloc[-1]
future_df["Rainfall"] = df["Rainfall"].iloc[-1]
future_df["Humidity"] = df["Humidity"].iloc[-1]

features = [
    "Households_Supplied",
    "Water_Per_Household_L",
    "Day",
    "Month",
    "Day_of_Week",
    "Temperature",
    "Rainfall",
    "Humidity"
]

future_X = future_df[features]

future_df["Predicted_Water_Demand_L"] = model.predict(
    future_X
)

st.dataframe(
    future_df[
        [
            "Date",
            "Households_Supplied",
            "Water_Per_Household_L",
            "Predicted_Water_Demand_L"
        ]
    ],
    use_container_width=True
)

# --------------------------------------------------
# FUTURE FORECAST GRAPH
# --------------------------------------------------

fig4, ax4 = plt.subplots(figsize=(12, 5))

ax4.plot(
    future_df["Date"],
    future_df["Predicted_Water_Demand_L"],
    marker="o"
)

ax4.set_title("7-Day Predicted Water Demand")
ax4.set_xlabel("Date")
ax4.set_ylabel("Predicted Water Demand (Litres)")
ax4.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig4)

# --------------------------------------------------
# INTERACTIVE WATER DEMAND PREDICTION
# --------------------------------------------------

st.header("🧮 Interactive Water Demand Prediction")

st.write(
    "Enter the number of households and water supplied "
    "per household to estimate total water demand."
)

col1, col2 = st.columns(2)

with col1:

    households = st.number_input(
        "Number of Households",
        min_value=1,
        max_value=125,
        value=63,
        step=1
    )

with col2:

    water_per_household = st.number_input(
        "Water per Household (Litres)",
        min_value=1,
        max_value=2000,
        value=400,
        step=50
    )

predict_button = st.button(
    "🔮 Predict Water Demand"
)

if predict_button:

    prediction_input = pd.DataFrame({
        "Households_Supplied": [households],
        "Water_Per_Household_L": [water_per_household],
        "Day": [df["Date"].iloc[-1].day],
        "Month": [df["Date"].iloc[-1].month],
        "Day_of_Week": [df["Date"].iloc[-1].dayofweek],
        "Temperature": [df["Temperature"].iloc[-1]],
        "Rainfall": [df["Rainfall"].iloc[-1]],
        "Humidity": [df["Humidity"].iloc[-1]]
    })

    prediction = model.predict(
        prediction_input[features]
    )[0]

    st.success(
        f"💧 Predicted Water Demand: "
        f"{prediction:,.0f} Litres"
    )

    # Simple calculated value for reference
    calculated_demand = (
        households * water_per_household
    )

    st.info(
        f"Reference calculation: "
        f"{households} households × "
        f"{water_per_household} L = "
        f"{calculated_demand:,.0f} L"
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.header("🤖 Machine Learning Model")

st.write(
    "The dashboard uses a Random Forest Regression model "
    "trained on household, water supply, calendar and weather features."
)

st.write("### Model Features")

st.write(
    """
    - Households Supplied
    - Water per Household
    - Day
    - Month
    - Day of Week
    - Temperature
    - Rainfall
    - Humidity
    """
)

# --------------------------------------------------
# IMPORTANT PROJECT NOTE
# --------------------------------------------------

st.warning(
    "Project Note: The current water-supply target values are "
    "based on the simulated 400 L per household baseline. "
    "For real-world forecasting, measured historical water "
    "consumption/supply data should be used."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.write("---")

st.caption(
    "Water Demand Forecasting Project | "
    "Shripalavan, Taluka Man, District Satara"
)
