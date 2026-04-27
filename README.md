# FairAI Decision Assistant

A modern AI-powered web application that predicts loan approvals while ensuring fairness and transparency.

## Features
- **User Input Panel**: Interactive sliders and inputs for applicant details.
- **Prediction System**: Uses Logistic Regression and Random Forest models.
- **Explainability**: Plain English translations of SHAP (SHapley Additive exPlanations) values.
- **AI Chat Assistant**: Mock chatbot to explain decisions, suggest improvements, and discuss fairness.
- **Counterfactuals**: Provides actionable suggestions to improve approval chances.
- **Bias Detection**: Compares predictions across demographic groups to flag potential historic bias.
- **AI Trust Score**: Calculates a confidence score based on model certainty, stability, and demographic parity.

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
