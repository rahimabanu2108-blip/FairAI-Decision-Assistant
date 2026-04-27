import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import shap
import copy

# --- Configuration ---
st.set_page_config(page_title="FairAI Decision Assistant", layout="wide", page_icon="🤖", initial_sidebar_state="collapsed")

# --- Custom CSS for Dark Theme & Cards ---
st.markdown("""
<style>
    /* Global Settings */
    body {
        color: #E0E0E0;
        background-color: #0E1117;
    }
    
    /* Card Container */
    .card {
        background-color: #1E2127;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
    }
    
    /* Headers */
    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: #00E5FF;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    
    /* Approved / Rejected styling */
    .status-approved {
        color: #00E676;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .status-rejected {
        color: #FF1744;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00E5FF;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0A0;
    }
    
    /* Trust Score Circular Bar Approximation */
    .trust-score-container {
        text-align: center;
        padding: 10px;
    }
    .trust-score-text {
        font-size: 3rem;
        font-weight: 900;
        color: #00E5FF;
    }
    
    /* Chat bubbles */
    .chat-user {
        background-color: #2b313c;
        padding: 10px 15px;
        border-radius: 15px 15px 0px 15px;
        margin-bottom: 10px;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .chat-bot {
        background-color: #1e4b59;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0px;
        margin-bottom: 10px;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    .chat-container {
        display: flow-root;
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        background: #15181e;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Generation ---
@st.cache_data
def load_data():
    """Generates a synthetic loan approval dataset with an injected gender bias."""
    np.random.seed(42)
    n_samples = 2000

    age = np.random.randint(22, 65, n_samples)
    income = np.random.normal(60000, 20000, n_samples).clip(20000, 150000)
    credit_score = np.random.normal(650, 80, n_samples).clip(300, 850)
    loan_amount = np.random.normal(15000, 5000, n_samples).clip(1000, 50000)
    gender = np.random.choice([0, 1], n_samples, p=[0.48, 0.52])
    education = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])

    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Education': education,
        'Gender': gender
    })

    base_score = (
        (df['Income'] / 150000) * 0.3 + 
        (df['Credit_Score'] / 850) * 0.5 - 
        (df['Loan_Amount'] / 50000) * 0.2 +
        (df['Education'] * 0.05)
    )

    # Bias: Males get an artificial bump
    base_score = np.where(df['Gender'] == 1, base_score + 0.1, base_score)
    prob_approval = 1 / (1 + np.exp(-(base_score - 0.4) * 10)) 
    approved = np.random.binomial(1, prob_approval)
    df['Approved'] = approved

    return df

# --- Model Training ---
@st.cache_resource
def train_models(df):
    X = df.drop('Approved', axis=1)
    y = df['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Calculate dataset bias for Trust Score
    y_pred = rf_model.predict(X_test)
    test_results = X_test.copy()
    test_results['Predicted'] = y_pred
    female_approval = test_results[test_results['Gender'] == 0]['Predicted'].mean()
    male_approval = test_results[test_results['Gender'] == 1]['Predicted'].mean()
    dp_diff = abs(male_approval - female_approval)

    return lr_model, rf_model, X_train, dp_diff

# --- Helper Logic Functions ---
def generate_english_explanation(shap_values, feature_names):
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0] # Take first instance if multiple
        
    contribs = list(zip(feature_names, shap_values))
    contribs.sort(key=lambda x: x[1], reverse=True)
    
    top_positive = [f for f in contribs if f[1] > 0]
    top_negative = [f for f in contribs if f[1] < 0]
    
    explanation = ""
    if top_positive:
        explanation += f"Your **{top_positive[0][0].replace('_', ' ')}** positively influenced the decision."
    if top_negative:
        # Get the largest negative magnitude (last in sorted list)
        explanation += f" However, your **{top_negative[-1][0].replace('_', ' ')}** reduced your approval chances."
        
    if not top_positive and not top_negative:
        explanation = "All factors were neutral."
        
    return explanation

def calculate_counterfactuals(model, input_data, current_prob):
    suggestions = []
    
    # Test increasing income
    cf_income = input_data.copy()
    cf_income['Income'] += 10000
    new_prob_inc = model.predict_proba(cf_income)[0][1]
    if new_prob_inc > current_prob + 0.02:
        suggestions.append({
            "action": "Increase Income by $10,000",
            "impact": new_prob_inc - current_prob
        })
        
    # Test decreasing loan amount
    cf_loan = input_data.copy()
    cf_loan['Loan_Amount'] = max(1000, cf_loan['Loan_Amount'].iloc[0] - 5000)
    new_prob_loan = model.predict_proba(cf_loan)[0][1]
    if new_prob_loan > current_prob + 0.02:
        suggestions.append({
            "action": "Reduce Loan Amount by $5,000",
            "impact": new_prob_loan - current_prob
        })
        
    # Test increasing credit score
    cf_credit = input_data.copy()
    cf_credit['Credit_Score'] = min(850, cf_credit['Credit_Score'].iloc[0] + 50)
    new_prob_credit = model.predict_proba(cf_credit)[0][1]
    if new_prob_credit > current_prob + 0.02:
        suggestions.append({
            "action": "Improve Credit Score by 50 points",
            "impact": new_prob_credit - current_prob
        })
        
    # Sort by highest impact
    suggestions.sort(key=lambda x: x['impact'], reverse=True)
    return suggestions[:2]

def calculate_trust_score(probability, dp_diff, model_type):
    # Confidence (distance from 0.5)
    confidence = abs(probability - 0.5) * 2 # 0 to 1
    
    # Bias penalty (if dp_diff is high, penalty is high)
    bias_penalty = min(1.0, dp_diff * 5) 
    
    # Base stability (RF generally more robust to outliers here)
    stability = 0.9 if model_type == "Random Forest" else 0.8
    
    score = (0.5 * confidence) + (0.3 * (1 - bias_penalty)) + (0.2 * stability)
    return max(0, min(100, int(score * 100)))

# --- Initialize App State ---
df = load_data()
lr_model, rf_model, X_train, dp_diff = train_models(df)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "content": "Hi! I am the FairAI Assistant. Feel free to ask me why you were approved/rejected, how to improve your chances, or if the decision is fair!"}
    ]
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = False

def reset_app():
    st.session_state.chat_history = [
        {"role": "bot", "content": "Hi! I am the FairAI Assistant. Feel free to ask me why you were approved/rejected, how to improve your chances, or if the decision is fair!"}
    ]
    st.session_state.reset_trigger = not st.session_state.reset_trigger

# --- Header ---
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.title("🤖 FairAI Decision Assistant")
    st.markdown("Predicting outcomes with *transparency* and *fairness* built-in.")
with col_head2:
    st.button("🔄 Reset Session", on_click=reset_app, use_container_width=True)

st.markdown("---")

# --- Main Layout ---
col_left, col_right = st.columns([1, 1.5])

# === LEFT COLUMN ===
with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>👤 Applicant Details</div>", unsafe_allow_html=True)
    
    model_choice = st.selectbox("Decision Engine", ["Random Forest", "Logistic Regression"], index=0)
    
    user_age = st.slider("Age", 18, 80, 35)
    user_income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=60000, step=5000)
    user_credit = st.slider("Credit Score", 300, 850, 680)
    user_loan = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
    
    edu_display = st.selectbox("Education Level", ["High School", "Bachelors", "Masters+"])
    edu_map = {"High School": 0, "Bachelors": 1, "Masters+": 2}
    user_edu = edu_map[edu_display]
    
    gender_display = st.radio("Gender", ["Female", "Male"], horizontal=True)
    user_gender = 0 if gender_display == "Female" else 1
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process Prediction Input
    input_data = pd.DataFrame({
        'Age': [user_age],
        'Income': [user_income],
        'Credit_Score': [user_credit],
        'Loan_Amount': [user_loan],
        'Education': [user_edu],
        'Gender': [user_gender]
    })

    model = rf_model if model_choice == "Random Forest" else lr_model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # SHAP logic
    if model_choice == "Random Forest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        shap_val_to_plot = shap_values[:, :, 1] if len(np.shape(shap_values)) == 3 else shap_values[1]
    else:
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(input_data)
        shap_val_to_plot = shap_values

    # Trust Score calculation
    trust_score = calculate_trust_score(probability, dp_diff, model_choice)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>🛡️ AI Trust Score</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='trust-score-container'><div class='trust-score-text'>{trust_score}%</div></div>", unsafe_allow_html=True)
    st.progress(trust_score / 100)
    st.caption("Based on Model Confidence, Demographic Parity Stability, and Decision Robustness.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Bias Badge
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>⚖️ Bias Detection</div>", unsafe_allow_html=True)
    if dp_diff > 0.05:
        st.warning("⚠️ Potential Bias Detected. Historic data exhibits gender disparity which the model may have learned.")
    else:
        st.success("✅ No significant bias detected.")
    
    # Small Bias bar chart
    fig, ax = plt.subplots(figsize=(4, 1.5))
    fig.patch.set_facecolor('#1E2127')
    ax.set_facecolor('#1E2127')
    sns.barplot(x=[0.4, dp_diff], y=['Threshold', 'Current Model'], palette=['#555', '#FF1744' if dp_diff > 0.05 else '#00E676'], ax=ax)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# === RIGHT COLUMN ===
with col_right:
    # Prediction Status
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>🎯 Decision Status</div>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown("<div class='status-approved'>✅ APPROVED</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-rejected'>❌ REJECTED</div>", unsafe_allow_html=True)
        
    st.progress(probability)
    st.markdown(f"<div style='text-align:center; color:#A0A0A0; margin-top:5px;'>Approval Probability: {probability:.1%}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Explainability
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>🧠 Explainability</div>", unsafe_allow_html=True)
    
    english_explanation = generate_english_explanation(shap_val_to_plot, input_data.columns.tolist())
    st.info(english_explanation)
    
    # Counterfactuals
    if prediction == 0 or probability < 0.8:
        suggestions = calculate_counterfactuals(model, input_data, probability)
        if suggestions:
            st.markdown("**💡 How to improve your chances:**")
            for sug in suggestions:
                st.markdown(f"- **{sug['action']}** (Estimated Impact: <span style='color:#00E676;'>+{sug['impact']:.1%}</span>)", unsafe_allow_html=True)
                
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Assistant
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>💬 AI Chat Assistant</div>", unsafe_allow_html=True)
    
    # Render Chat History
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    def handle_chat():
        user_input = st.session_state.chat_input
        if not user_input.strip(): return
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Rule-based logic
        lower_input = user_input.lower()
        if "why" in lower_input or "explain" in lower_input:
            response = f"Based on the AI's analysis: {english_explanation} The model looks at a combination of these factors to determine reliability."
        elif "improve" in lower_input or "better" in lower_input or "chances" in lower_input:
            sugs = calculate_counterfactuals(model, input_data, probability)
            if sugs:
                sug_text = ", or ".join([s['action'] for s in sugs])
                response = f"To improve your chances, the model suggests you could: {sug_text}."
            else:
                response = "Your current profile is very strong. Significant improvements would require large changes to income or credit score."
        elif "fair" in lower_input or "bias" in lower_input:
            if dp_diff > 0.05:
                response = "Our bias detection tools noticed a historic imbalance in the training data related to gender. While we use advanced algorithms, you should be aware that the system Trust Score reflects this potential bias."
            else:
                response = "Yes! Our continuous monitoring shows no significant bias across demographic groups for this decision engine."
        else:
            response = "I'm a simulated assistant for this demo. Try asking me 'Why was I approved?', 'How can I improve?', or 'Is this fair?'"
            
        st.session_state.chat_history.append({"role": "bot", "content": response})
        st.session_state.chat_input = "" # clear input

    st.text_input("Ask a question...", key="chat_input", on_change=handle_chat, placeholder="e.g. Why was I rejected?")
    
    st.markdown("</div>", unsafe_allow_html=True)
