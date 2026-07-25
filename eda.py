import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Page Architecture & Design Configuration
st.set_page_config(page_title="CredSecure Pro: 4-Stage Analytics Engine", layout="wide")
st.title("🏦 CredSecure Pro: End-to-End Credit Risk Framework")
st.markdown("A self-contained portfolio project demonstrating the four pillars of data decision frameworks: Descriptive, Diagnostic, Predictive, and Prescriptive Analytics.")

# 2. Synthetic Credit Data Generator Engine
@st.cache_data
def load_historical_credit_data(records=400):
    np.random.seed(42)
    
    credit_scores = np.random.randint(300, 850, records)
    income_annual = np.random.uniform(25000, 160000, records).round(0)
    debt_to_income = np.random.uniform(0.1, 0.70, records).round(2)
    employment_years = np.random.randint(0, 20, records)
    loan_amount = (income_annual * np.random.uniform(0.5, 3.0, records)).round(0)
    
    # Advanced feature combination establishing ground truth
    risk_signal = ((850 - credit_scores) / 550) * 1.8 + (debt_to_income * 2.2) - (employment_years / 25) + (loan_amount / income_annual * 0.5)
    default_status = np.where(risk_signal > 1.6, 1, 0) # 1 = Defaulted, 0 = Fully Paid

    data = {
        'CustomerID': [f"CRD-{i:05d}" for i in range(1, records + 1)],
        'CreditScore': credit_scores,
        'AnnualIncome_USD': income_annual,
        'DebtToIncome_Ratio': debt_to_income,
        'EmploymentHistory_Years': employment_years,
        'RequestedLoan_USD': loan_amount,
        'LoanStatus': default_status 
    }
    return pd.DataFrame(data)

df = load_historical_credit_data()

# Create display variations for visualization clarity
display_df = df.copy()
display_df['LoanStatus_Label'] = display_df['LoanStatus'].map({1: 'Defaulted', 0: 'Fully Paid'})

# 3. Structural Modeling Framework Setup (Predictive Pre-requisite)
X = df[['CreditScore', 'AnnualIncome_USD', 'DebtToIncome_Ratio', 'EmploymentHistory_Years', 'RequestedLoan_USD']]
y = df['LoanStatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- SIDEBAR: INTERACTIVE APPLICATION UNDERWRITING ---
st.sidebar.header("🎯 Live Applicant Underwriting")
st.sidebar.markdown("Input prospective variables to score an individual applicant instantly via the active backend ML model.")

in_score = st.sidebar.slider("Credit Score Assessment", 300, 850, 680)
in_income = st.sidebar.number_input("Annual Gross Income ($)", value=60000, step=5000)
in_dti = st.sidebar.slider("Current Debt-to-Income (DTI)", 0.0, 1.0, 0.35)
in_emp = st.sidebar.slider("Employment Stability (Years)", 0, 20, 4)
in_loan = st.sidebar.number_input("Requested Principal Value ($)", value=45000, step=5000)

# --- FOUR STAGES OF ANALYTICS NAVIGATION TABS ---
tab_descriptive, tab_diagnostic, tab_predictive, tab_prescriptive = st.tabs([
    "📊 1. Descriptive (What Happened?)", 
    "🔍 2. Diagnostic (Why Did It Happen?)", 
    "🤖 3. Predictive (What Will Happen?)", 
    "📈 4. Prescriptive (How Can We Optimize?)"
])

# ==============================================================================
# TAB 1: DESCRIPTIVE ANALYTICS
# ==============================================================================
with tab_descriptive:
    st.header("Portfolio Descriptive Analytics Overview")
    st.markdown("Aggregating historical loan accounts to establish baseline institutional KPIs.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Historical Audited Records", len(df))
    with col2:
        default_rate = (df['LoanStatus'] == 1).mean() * 100
        st.metric("Portfolio Default Rate", f"{default_rate:.1f}%")
    with col3:
        st.metric("Mean Portfolio Credit Score", int(df['CreditScore'].mean()))
    with col4:
        st.metric("Mean Debt-to-Income Ratio", f"{df['DebtToIncome_Ratio'].mean():.2f}")
        
    st.markdown("---")
    st.subheader("Historical Risk Master Ledger File")
    ledger_filter = st.radio("Isolate Records:", ["Show All Records", "Show Defaults Only", "Show Fully Paid Only"], horizontal=True)
    
    if ledger_filter == "Show Defaults Only":
        st.dataframe(display_df[display_df['LoanStatus'] == 1], use_container_width=True)
    elif ledger_filter == "Show Fully Paid Only":
        st.dataframe(display_df[display_df['LoanStatus'] == 0], use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)

# ==============================================================================
# TAB 2: DIAGNOSTIC ANALYTICS
# ==============================================================================
with tab_diagnostic:
    st.header("Root Cause Diagnostic Exploratory Sandbox")
    st.markdown("Analyzing variances, feature dependencies, and distributions to isolate risk drivers.")
    
    selected_feature = st.selectbox("Select Feature Distribution Profile to Analyze:", ['CreditScore', 'DebtToIncome_Ratio', 'RequestedLoan_USD'])
    
    fig_hist = px.histogram(display_df, x=selected_feature, color="LoanStatus_Label", barmode="overlay",
                            title=f"Population Variance: {selected_feature} Grouped By Repayment Status",
                            color_discrete_map={'Fully Paid': '#2ECC71', 'Defaulted': '#E74C3C'}, marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_scatter = px.scatter(display_df, x="CreditScore", y="DebtToIncome_Ratio", color="LoanStatus_Label",
                                 title="Bivariate Clustering: Credit Score vs Debt-to-Income",
                                 color_discrete_map={'Fully Paid': '#2ECC71', 'Defaulted': '#E74C3C'})
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_d2:
        corr_matrix = df[['CreditScore', 'AnnualIncome_USD', 'DebtToIncome_Ratio', 'RequestedLoan_USD', 'EmploymentHistory_Years']].corr().round(2)
        fig_heat = ff.create_annotated_heatmap(z=corr_matrix.values, x=list(corr_matrix.columns), y=list(corr_matrix.index), colorscale='RdBu_r')
        fig_heat.update_layout(title="Linear Feature Dependency Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================================
# TAB 3: PREDICTIVE ANALYTICS
# ==============================================================================
with tab_predictive:
    st.header("Algorithmic Machine Learning Sandbox")
    st.markdown("Training functional mathematical models to score unseen credit risk profiles.")
    
    col_ml1, col_ml2 = st.columns([1, 2])
    
    with col_ml1:
        st.subheader("Model Framework Settings")
        selected_model = st.radio("Choose Backend Classifier:", ["Logistic Regression", "Random Forest"])
        
        if selected_model == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            trees = st.slider("Number of Decision Trees:", 10, 100, 50, step=10)
            model = RandomForestClassifier(n_estimators=trees, random_state=42)
            
        # Execute instant training block
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        
        st.metric("Live Testing Accuracy Score", f"{acc*100:.1f}%")
        
    with col_ml2:
        st.subheader("Confusion Matrix Matrix Blueprint")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(z=cm, x=["Predicted Paid", "Predicted Default"], y=["Actual Paid", "Actual Default"], colorscale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Algorithmic Feature Significance Mapping")
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.DataFrame({'Feature Element': X.columns, 'Weight Contribution': model.feature_importances_}).sort_values(by='Weight Contribution', ascending=False)
        fig_imp = px.bar(feat_imp, x='Weight Contribution', y='Feature Element', orientation='h', title="Random Forest Information Gain Distribution")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("💡 **Regression Insight:** Logistic Regression evaluates structural variables via mathematical log-odds coefficients instead of feature importance trees. Switch to Random Forest to map informational weights.")

    # Execute Sidebar updates based on the currently trained pipeline state
    applicant_features = pd.DataFrame([[in_score, in_income, in_dti, in_emp, in_loan]], columns=X.columns)
    app_prob = model.predict_proba(applicant_features)[0, 1]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Calculated Output Metrics")
    if app_prob < 0.30:
        st.sidebar.success(f"🟢 Approved Profile: {app_prob*100:.1f}% Risk")
    elif app_prob < 0.60:
        st.sidebar.warning(f"🟡 Conditional Manual Review: {app_prob*100:.1f}% Risk")
    else:
        st.sidebar.error(f"🔴 Auto-Rejection Triggered: {app_prob*100:.1f}% Risk")
        
    # Explainable AI (XAI) Micro-Driver Alerts
    drivers = []
    if in_score < 580: drivers.append("Depressed Bureau Credit Rating")
    if in_dti > 0.48: drivers.append("Extreme Over-Leverage Risk (High DTI)")
    if in_loan > (in_income * 2.2): drivers.append("Disproportionate Principal Exposure Balance")
    if drivers:
        st.sidebar.warning("⚠️ **Primary Risk Drivers Identified:**\n" + "\n".join([f"- {d}" for d in drivers]))

# ==============================================================================
# TAB 4: PRESCRIPTIVE ANALYTICS
# ==============================================================================
with tab_prescriptive:
    st.header("Financial Decision Optimization Simulator")
    st.markdown("Translating analytical probabilities into actionable corporate policy boundaries.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        avg_revenue = st.number_input("Average Expected Value Earned Per Approved Clean Loan ($)", value=9500)
    with col_p2:
        avg_loss = st.number_input("Average Write-Off Loss Realized Per Default Event ($)", value=30000)
        
    st.markdown("---")
    st.subheader("Simulate Decision Boundary Optimization")
    st.markdown("Adjust the risk allowance slider below. The tool evaluates the probabilities of the test set to isolate the precise cutoff point that maximizes overall net yields.")
    
    threshold = st.slider("Maximum Permissible Model Probability Cap for Approvals", 0.05, 0.95, 0.40, step=0.05)
    
    # Calculate performance over test sets using current slider constraints
    sim_df = pd.DataFrame({'Actual': y_test, 'Calculated_Prob': y_probs})
    sim_df['Action'] = np.where(sim_df['Calculated_Prob'] <= threshold, 'Approve', 'Reject')
    
    true_approved = len(sim_df[(sim_df['Action'] == 'Approve') & (sim_df['Actual'] == 0)])
    false_approved = len(sim_df[(sim_df['Action'] == 'Approve') & (sim_df['Actual'] == 1)])
    
    gross_yield = true_approved * avg_revenue
    loss_penalties = false_approved * avg_loss
    net_portfolio_value = gross_yield - loss_penalties
    
    # Render Financial Projections Metrics
    cm1, cm2, cm3, cm4 = st.columns(4)
    with cm1:
        st.metric("Total Automated Approvals", true_approved + false_approved)
    with cm2:
        st.metric("Profitable Assets Maintained", true_approved)
    with cm3:
        st.metric("Bad Debt Leaked Into Portfolio", false_approved, delta=f"{false_approved} defaults", delta_color="inverse")
    with cm4:
        st.metric("Net Projected Yield Simulation", f"${net_portfolio_value:,.2f}")
        
    st.info("💡 **Prescriptive Takeaway:** Notice how raising the threshold increases total approvals but also invites heavy write-off losses. This simulation directly provides the most profitable policy setting for the risk department.")