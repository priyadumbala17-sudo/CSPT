import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. DATASET GENERATION & PREPROCESSING
# ==========================================
@st.cache_data
def load_data(n_samples=891):
    pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(['male', 'female'], size=n_samples, p=[0.64, 0.36])
    age = np.random.normal(29, 14, size=n_samples)
    age[age < 1] = 1
    
    # Simulate missing age values
    missing_age = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    age[missing_age] = np.nan
    
    sibsp = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.68, 0.23, 0.05, 0.02, 0.02])
    parch = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.76, 0.13, 0.09, 0.02])
    fare = (4 - pclass) * 25 + np.random.exponential(15, size=n_samples)
    embarked = np.random.choice(['S', 'C', 'Q'], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Nonlinear survival probability logic
    prob = (
        0.45 * (sex == 'female') +
        0.25 * (pclass == 1) +
        0.15 * (np.nan_to_num(age, nan=30) < 12) +
        0.10 * ((sibsp + parch + 1) >= 2) * ((sibsp + parch + 1) <= 4)
    )
    survived = np.random.binomial(1, np.clip(prob, 0.05, 0.95))

    df = pd.DataFrame({
        'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp,
        'Parch': parch, 'Fare': fare, 'Embarked': embarked, 'Survived': survived
    })
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

@st.cache_resource
def train_models(df, max_depth, n_estimators):
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    num_cols = ['Age', 'Fare', 'FamilySize', 'IsAlone']
    cat_cols = ['Pclass', 'Sex', 'Embarked']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ]), cat_cols)
    ])

    # 1. Decision Tree Pipeline
    dt_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, random_state=42))
    ])

    # 2. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=3, random_state=42))
    ])

    dt_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # Feature names output
    cat_encoder = rf_pipeline.named_steps['prep'].named_transformers_['cat'].named_steps['encoder']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_cols

    return dt_pipeline, rf_pipeline, X_test, y_test, all_feature_names

# ==========================================
# 2. STREAMLIT UI LAYOUT
# ==========================================

st.title("🚢 Titanic Survival Analysis & Prediction")
st.write("Compare **Decision Tree** vs. **Random Forest** models and analyze feature importances.")

df = load_data()

# Sidebar - Controls & Inputs
st.sidebar.header("⚙️ Model Hyperparameters")
max_depth = st.sidebar.slider("Tree Max Depth", min_value=2, max_value=12, value=4)
n_estimators = st.sidebar.slider("Random Forest Trees", min_value=10, max_value=200, value=100, step=10)

dt_pipeline, rf_pipeline, X_test, y_test, feature_names = train_models(df, max_depth, n_estimators)

st.sidebar.markdown("---")
st.sidebar.header("📋 Passenger Input")
pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.sidebar.radio("Sex", ['female', 'male'])
age = st.sidebar.slider("Age", 1, 80, 28)
sibsp = st.sidebar.number_input("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents / Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid ($)", 0, 300, 32)
embarked = st.sidebar.selectbox("Port of Embarkation", ['S', 'C', 'Q'], format_func={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}.get)

model_choice = st.sidebar.radio("Select Prediction Model", ["Random Forest", "Decision Tree"])

# Predict Button logic
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

passenger_df = pd.DataFrame([{
    'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp,
    'Parch': parch, 'Fare': fare, 'Embarked': embarked,
    'FamilySize': family_size, 'IsAlone': is_alone
}])

selected_model = rf_pipeline if model_choice == "Random Forest" else dt_pipeline
prob = selected_model.predict_proba(passenger_df)[0][1]
pred = selected_model.predict(passenger_df)[0]

# Main Area Split
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Live Prediction Outcome")
    if pred == 1:
        st.success(f"### Outcome: Survived 🎉\n**Estimated Survival Probability:** `{prob * 100:.1f}%`")
    else:
        st.error(f"### Outcome: Perished ⚠️\n**Estimated Survival Probability:** `{prob * 100:.1f}%`")

    # Metrics Summary
    dt_acc = accuracy_score(y_test, dt_pipeline.predict(X_test))
    rf_acc = accuracy_score(y_test, rf_pipeline.predict(X_test))

    st.markdown("### 📊 Test Accuracy Comparison")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Decision Tree Accuracy", f"{dt_acc * 100:.2f}%")
    m_col2.metric("Random Forest Accuracy", f"{rf_acc * 100:.2f}%")

with col2:
    st.subheader("📌 Feature Importance Analysis")
    rf_clf = rf_pipeline.named_steps['clf']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='crest', ax=ax)
    ax.set_title("Random Forest Gini Importances")
    st.pyplot(fig)

st.markdown("---")

# Feature Insights Tab
st.subheader("🔍 Deep-Dive Model Inspection")
tab1, tab2, tab3 = st.tabs(["Decision Tree Visualization", "Dataset Preview", "Confusion Matrix"])

with tab1:
    st.write("#### Pruned Decision Tree Structure")
    fig_tree, ax_tree = plt.subplots(figsize=(16, 6))
    plot_tree(
        dt_pipeline.named_steps['clf'],
        feature_names=feature_names,
        class_names=['Perished', 'Survived'],
        filled=True,
        rounded=True,
        ax=ax_tree,
        fontsize=8
    )
    st.pyplot(fig_tree)

with tab2:
    st.dataframe(df.head(15), use_container_width=True)

with tab3:
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        st.write("**Decision Tree Matrix**")
        cm_dt = confusion_matrix(y_test, dt_pipeline.predict(X_test))
        fig_cm1, ax_cm1 = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax_cm1)
        ax_cm1.set_xlabel("Predicted")
        ax_cm1.set_ylabel("Actual")
        st.pyplot(fig_cm1)

    with c_col2:
        st.write("**Random Forest Matrix**")
        cm_rf = confusion_matrix(y_test, rf_pipeline.predict(X_test))
        fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax_cm2)
        ax_cm2.set_xlabel("Predicted")
        ax_cm2.set_ylabel("Actual")
        st.pyplot(fig_cm2)