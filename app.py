# =========================================================
# LAPTOP MARKET ANALYSIS DASHBOARD
# STREAMLIT + MACHINE LEARNING
# =========================================================

# RUN:
# streamlit run app.py

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Laptop Analytics Dashboard",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop Market Analysis Dashboard")

st.markdown(
    "Interactive Dashboard using Streamlit + Machine Learning"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel("AMAZON_LAPTOP_DATASET_EXCEL_CLEANED.xlsx")

    return df


df = load_data()


# =========================================================
# DATA CLEANING
# =========================================================

# PRICE CLEANING
df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)


# RATING CLEANING
df["Rating"] = (
    df["Rating"]
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)


# RAM CLEANING
df["RAM"] = (
    df["RAM"]
    .astype(str)
    .str.extract(r'(\d+)')[0]
)

df["RAM"] = pd.to_numeric(
    df["RAM"],
    errors="coerce"
)


# SSD CLEANING
df["SSD"] = (
    df["SSD"]
    .astype(str)
    .str.extract(r'(\d+)')[0]
)

df["SSD"] = pd.to_numeric(
    df["SSD"],
    errors="coerce"
)


# FILL NULL VALUES
df["Price"] = df["Price"].fillna(df["Price"].median())
df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["RAM"] = df["RAM"].fillna(df["RAM"].median())
df["SSD"] = df["SSD"].fillna(df["SSD"].median())


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔍 Filters")

brand_filter = st.sidebar.multiselect(
    "Select Brand",
    options=df["Brand"].unique(),
    default=df["Brand"].unique()
)

ram_filter = st.sidebar.multiselect(
    "Select RAM",
    options=sorted(df["RAM"].unique()),
    default=sorted(df["RAM"].unique())
)

os_filter = st.sidebar.multiselect(
    "Operating System",
    options=df["Operating System"].unique(),
    default=df["Operating System"].unique()
)

price_filter = st.sidebar.slider(
    "Price Range",
    int(df["Price"].min()),
    int(df["Price"].max()),
    (
        int(df["Price"].min()),
        int(df["Price"].max())
    )
)


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["Brand"].isin(brand_filter)) &
    (df["RAM"].isin(ram_filter)) &
    (df["Operating System"].isin(os_filter)) &
    (df["Price"] >= price_filter[0]) &
    (df["Price"] <= price_filter[1])
]


# =========================================================
# KPI SECTION
# =========================================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Laptops",
    len(filtered_df)
)

col2.metric(
    "Average Price",
    f"₹ {round(filtered_df['Price'].mean(), 2)}"
)

col3.metric(
    "Average Rating",
    round(filtered_df["Rating"].mean(), 2)
)

col4.metric(
    "Brands",
    filtered_df["Brand"].nunique()
)


# =========================================================
# BRAND DISTRIBUTION
# =========================================================

st.subheader("🏆 Top Laptop Brands")

brand_counts = (
    filtered_df["Brand"]
    .value_counts()
    .head(10)
)

fig_brand = px.bar(
    x=brand_counts.index,
    y=brand_counts.values,
    labels={
        "x": "Brand",
        "y": "Count"
    },
    title="Top Laptop Brands"
)

st.plotly_chart(
    fig_brand,
    use_container_width=True
)


# =========================================================
# PRICE VS RAM
# =========================================================

st.subheader("💾 RAM vs Price")

fig_ram = px.scatter(
    filtered_df,
    x="RAM",
    y="Price",
    color="Brand",
    size="SSD",
    hover_data=["Processor"],
    title="RAM vs Price"
)

st.plotly_chart(
    fig_ram,
    use_container_width=True
)


# =========================================================
# SSD DISTRIBUTION
# =========================================================

st.subheader("📊 SSD Distribution")

fig_ssd = px.histogram(
    filtered_df,
    x="SSD",
    nbins=20,
    title="SSD Storage Distribution"
)

st.plotly_chart(
    fig_ssd,
    use_container_width=True
)


# =========================================================
# OPERATING SYSTEM DISTRIBUTION
# =========================================================

st.subheader("🖥 Operating System Distribution")

os_counts = filtered_df["Operating System"].value_counts()

fig_os = px.pie(
    values=os_counts.values,
    names=os_counts.index,
    title="Operating System Share"
)

st.plotly_chart(
    fig_os,
    use_container_width=True
)


# =========================================================
# CORRELATION HEATMAP
# =========================================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=np.number)

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)


# =========================================================
# MACHINE LEARNING SECTION
# =========================================================

st.subheader("🤖 Laptop Price Prediction")


# LABEL ENCODING
encoder_brand = LabelEncoder()
encoder_processor = LabelEncoder()
encoder_graphics = LabelEncoder()
encoder_os = LabelEncoder()

ml_df = filtered_df.copy()

ml_df["Brand"] = encoder_brand.fit_transform(
    ml_df["Brand"]
)

ml_df["Processor"] = encoder_processor.fit_transform(
    ml_df["Processor"]
)

ml_df["Graphics Card"] = encoder_graphics.fit_transform(
    ml_df["Graphics Card"]
)

ml_df["Operating System"] = encoder_os.fit_transform(
    ml_df["Operating System"]
)


# FEATURES
X = ml_df[
    [
        "Rating",
        "RAM",
        "SSD",
        "Brand",
        "Processor",
        "Graphics Card",
        "Operating System"
    ]
]

y = ml_df["Price"]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# MODEL
model = RandomForestRegressor()

model.fit(X_train, y_train)


# PREDICTION
predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

st.success(
    f"Model Accuracy (R² Score): {round(score, 2)}"
)


# =========================================================
# PREDICTION UI
# =========================================================

st.subheader("🔮 Predict Laptop Price")

input_rating = st.slider(
    "Rating",
    1.0,
    5.0,
    4.0
)

input_ram = st.selectbox(
    "RAM",
    sorted(df["RAM"].unique())
)

input_ssd = st.selectbox(
    "SSD",
    sorted(df["SSD"].unique())
)

input_brand = st.selectbox(
    "Brand",
    df["Brand"].unique()
)

input_processor = st.selectbox(
    "Processor",
    df["Processor"].unique()
)

input_graphics = st.selectbox(
    "Graphics Card",
    df["Graphics Card"].unique()
)

input_os = st.selectbox(
    "Operating System",
    df["Operating System"].unique()
)


# ENCODE INPUTS
brand_encoded = encoder_brand.transform([input_brand])[0]

processor_encoded = encoder_processor.transform([input_processor])[0]

graphics_encoded = encoder_graphics.transform([input_graphics])[0]

os_encoded = encoder_os.transform([input_os])[0]


# FINAL PREDICTION
predicted_price = model.predict([[
    input_rating,
    input_ram,
    input_ssd,
    brand_encoded,
    processor_encoded,
    graphics_encoded,
    os_encoded
]])[0]


st.success(
    f"💰 Predicted Laptop Price: ₹ {round(predicted_price, 2)}"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("📌 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig_importance = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Feature Importance"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.subheader("📖 Business Insights")

avg_price = filtered_df["Price"].mean()

high_ram = filtered_df[
    filtered_df["RAM"] >= 16
]

premium = filtered_df[
    filtered_df["Price"] > avg_price
]

st.info(f"""

✅ Average Laptop Price: ₹ {round(avg_price, 2)}

✅ High RAM laptops count: {len(high_ram)}

✅ Premium laptops count: {len(premium)}

✅ Gaming laptops with dedicated graphics cards are significantly more expensive.

✅ SSD size strongly impacts pricing.

✅ Premium brands dominate high-price segments.

✅ Higher RAM configurations generally receive better ratings.

""")


# =========================================================
# DOWNLOAD DATA
# =========================================================

# st.subheader("⬇ Download Filtered Data")

# csv = filtered_df.to_csv(index=False)

# st.download_button(
    # label="Download CSV",
    # data=csv,
    # file_name="filtered_laptops.csv",
    # mime="text/csv"
# )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "🚀 Developed using Streamlit + Machine Learning"
)

st.markdown(
    "🚀 Developed Ramchandra Suryawanshi"
)