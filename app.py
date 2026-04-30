
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="FMCG Demand Forecasting Portal", layout="wide")

# -----------------------------
# CUSTOM CSS (Dark Modern Style)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
        color: white;
    }
    .stApp {
        background-color: #0f1117;
        color: white;
    }
    h1, h2, h3, h4 {
        color: white;
    }
    .metric-card {
        background-color: #1c1f2b;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("📦 FMCG Demand Forecasting Portal")
st.markdown("An interactive business intelligence and demand prediction system for FMCG supply chains.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation Portal")
page = st.sidebar.radio("Go to Section", [
    "Dashboard Home",
    "Dataset Preview",
    "Sales Trend",
    "Forecasting Model",
    "Future Forecast",
    "Demand Prediction"
])

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("krishanukalita/fmcg-sales-demand-forecasting-and-optimization")
    files = os.listdir(path)
    file_path = os.path.join(path, files[0])

    df = pd.read_csv(file_path)

    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Detect possible date and sales columns
    possible_date_cols = [col for col in df.columns if 'date' in col or 'day' in col or 'time' in col]
    possible_sales_cols = [col for col in df.columns if 'sale' in col or 'revenue' in col or 'demand' in col or 'quantity' in col or 'units' in col]

    date_col = possible_date_cols[0]
    sales_col = possible_sales_cols[0]

    # Convert date
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Group sales
    df_grouped = df.groupby(date_col)[sales_col].sum().reset_index()
    df_grouped.columns = ['date', 'sales']

    return df, df_grouped

df, df_grouped = load_data()

# -----------------------------
# MODEL BUILDING
# -----------------------------
df_grouped['t'] = np.arange(len(df_grouped))

train_size = int(len(df_grouped) * 0.8)
train = df_grouped[:train_size]
test = df_grouped[train_size:].copy()

model = LinearRegression()
model.fit(train[['t']], train['sales'])

test['prediction'] = model.predict(test[['t']])

# Metrics
mae = mean_absolute_error(test['sales'], test['prediction'])
rmse = np.sqrt(mean_squared_error(test['sales'], test['prediction']))
total_sales = int(df_grouped['sales'].sum())
avg_sales = int(df_grouped['sales'].mean())

# Future forecast setup
forecast_days = st.sidebar.slider("📅 Select Days to Forecast", 7, 60, 30)
future_t = np.arange(len(df_grouped), len(df_grouped) + forecast_days).reshape(-1, 1)
future_pred = model.predict(future_t)

future_dates = pd.date_range(start=df_grouped['date'].max() + pd.Timedelta(days=1), periods=forecast_days)
future_df = pd.DataFrame({
    'date': future_dates,
    'forecast': future_pred
})

# -----------------------------
# DASHBOARD HOME
# -----------------------------
if page == "Dashboard Home":
    st.subheader("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"{total_sales:,}")
    col2.metric("Average Sales", f"{avg_sales:,}")
    col3.metric("Forecast Days", forecast_days)

    st.markdown("---")
    st.write("### Welcome to the FMCG Forecasting Portal")
    st.write("""
    This portal helps in:
    - Understanding historical sales patterns
    - Comparing actual and predicted sales
    - Forecasting future demand
    - Simulating business input conditions for demand estimation
    """)

# -----------------------------
# DATASET PREVIEW
# -----------------------------
elif page == "Dataset Preview":
    st.subheader("📂 Dataset Preview")

    st.write("### Raw Dataset")
    st.dataframe(df.head())

    st.write("### Grouped Daily Sales Data")
    st.dataframe(df_grouped.head())

    st.write("### Dataset Information")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

# -----------------------------
# SALES TREND
# -----------------------------
elif page == "Sales Trend":
    st.subheader("📈 Sales Trend Analysis")

    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df_grouped['date'], df_grouped['sales'], color='cyan')
    ax1.set_title("Sales Trend Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")
    ax1.grid(True)
    st.pyplot(fig1)

    st.info("This chart shows how FMCG sales fluctuate over time, helping identify trends and planning needs.")

# -----------------------------
# FORECASTING MODEL
# -----------------------------
elif page == "Forecasting Model":
    st.subheader("📉 Actual vs Predicted Sales")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(test['date'], test['sales'], label="Actual Sales", color='lime')
    ax2.plot(test['date'], test['prediction'], label="Predicted Sales", color='orange')
    ax2.set_title("Actual vs Predicted Sales")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.write("### Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("MAE", round(mae, 2))
    c2.metric("RMSE", round(rmse, 2))

    st.info("These metrics evaluate how accurately the forecasting model predicts demand.")

# -----------------------------
# FUTURE FORECAST
# -----------------------------
elif page == "Future Forecast":
    st.subheader("🔮 Future Demand Forecast")

    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(df_grouped['date'], df_grouped['sales'], label="Historical Sales", color='cyan')
    ax3.plot(future_df['date'], future_df['forecast'], label="Future Forecast", linestyle="dashed", color='magenta')
    ax3.set_title("Future Demand Forecast")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Sales")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    st.write("### Forecasted Data")
    st.dataframe(future_df.head(15))

    st.info("This section projects expected demand for future time periods to support inventory and supply chain planning.")

# -----------------------------
# DEMAND PREDICTION INPUT FORM
# -----------------------------
elif page == "Demand Prediction":
    st.subheader("📥 Input Features")
    st.write("Enter the business variables below to estimate expected demand.")

    # Input fields
    price = st.number_input("Price", min_value=0.0, value=50.0, step=1.0)
    discount = st.number_input("Discount (%)", min_value=0.0, value=10.0, step=1.0)
    inventory = st.number_input("Inventory Level", min_value=0, value=100, step=1)
    promotion = st.selectbox("Promotion", [0, 1], help="0 = No Promotion, 1 = Promotion Running")
    competitor_price = st.number_input("Competitor Price", min_value=0.0, value=50.0, step=1.0)
    category = st.selectbox("Category", ["Clothing", "Food", "Beverages", "Electronics", "Household"])

    st.markdown("---")

    if st.button("🔮 Predict Demand"):
        # Demo prediction logic
        predicted_demand = (
            500
            - (price * 2)
            + (discount * 5)
            + (inventory * 0.3)
            + (promotion * 100)
            + (competitor_price * 1.5)
        )

        category_factor = {
            "Clothing": 50,
            "Food": 120,
            "Beverages": 90,
            "Electronics": 70,
            "Household": 60
        }

        predicted_demand += category_factor[category]

        st.success(f"📦 Predicted Demand: {int(predicted_demand)} units")

        if predicted_demand > 700:
            st.info("📈 High Demand Expected — Maintain higher inventory levels.")
        elif predicted_demand > 500:
            st.info("📊 Moderate Demand Expected — Maintain balanced stock.")
        else:
            st.warning("📉 Low Demand Expected — Avoid overstocking.")

        # Optional mini bar chart
        chart_df = pd.DataFrame({
            "Feature": ["Price", "Discount", "Inventory", "Promotion", "Competitor Price"],
            "Value": [price, discount, inventory, promotion, competitor_price]
        })

        st.write("### Input Feature Summary")
        st.bar_chart(chart_df.set_index("Feature"))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("✅ **FMCG Demand Forecasting Portal Developed Using Streamlit**")