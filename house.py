import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set Page Config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. DATASET GENERATION & MODEL TRAINING
# ==========================================
@st.cache_data
def generate_data(n_samples=1000):
    sqft = np.random.randint(800, 4500, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], size=n_samples)
    age = np.random.randint(0, 50, size=n_samples)
    location = np.random.choice(['Suburbs', 'Downtown', 'Rural'], size=n_samples, p=[0.5, 0.3, 0.2])
    
    loc_multiplier = {'Suburbs': 1.2, 'Downtown': 1.6, 'Rural': 0.8}
    loc_factor = np.vectorize(loc_multiplier.get)(location)
    
    price = (
        (sqft * 150) +
        (bedrooms * 10000) +
        (bathrooms * 15000) -
        (age * 1200) + 50000
    ) * loc_factor + np.random.normal(0, 25000, size=n_samples)
    
    return pd.DataFrame({
        'SquareFeet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'Location': location,
        'Price': np.round(price, 2)
    })

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Price'])
    y = df['Price']

    num_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    cat_features = ['Location']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first'), cat_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return pipeline, metrics, X_test, y_test, y_pred

# Load data and model
df = generate_data()
model_pipeline, metrics, X_test, y_test, y_pred = train_model(df)

# ==========================================
# 2. STREAMLIT UI LAYOUT
# ==========================================

st.title("🏠 House Price Prediction Dashboard")
st.write("Predict estimated home market values using a **Linear Regression** model.")

st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("📋 House Specifications")

sqft = st.sidebar.number_input("Square Feet", min_value=500, max_value=10000, value=2200, step=50)
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=8, value=3)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)
age = st.sidebar.slider("Property Age (Years)", min_value=0, max_value=100, value=10)
location = st.sidebar.selectbox("Location / Neighborhood", options=['Suburbs', 'Downtown', 'Rural'])

# Predict Button logic
input_data = pd.DataFrame([{
    'SquareFeet': sqft,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age': age,
    'Location': location
}])

predicted_price = model_pipeline.predict(input_data)[0]

# Main Area Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💵 Valuation Result")
    st.metric(
        label="Estimated House Price",
        value=f"${predicted_price:,.2f}"
    )

    st.markdown("### 📊 Model Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("R² Score", f"{metrics['r2']:.3f}")
    m_col2.metric("MAE", f"${metrics['mae']:,.0f}")
    m_col3.metric("RMSE", f"${metrics['rmse']:,.0f}")

with col2:
    st.subheader("📈 Predicted vs. Actual Prices")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax, color='#1f77b4')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

st.markdown("---")

# Feature Insights Tab
st.subheader("🔎 Dataset Overview & Features")
tab1, tab2 = st.tabs(["Dataset Preview", "Feature Correlation"])

with tab1:
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
    num_df = df.select_dtypes(include=[np.number])
    sns.heatmap(num_df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)