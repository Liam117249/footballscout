import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# ==========================================
# 1. APP CONFIGURATION & CLUSTER NAMES
# ==========================================
st.set_page_config(page_title="FIFA Scout Pro", layout="wide")

# Cluster descriptions
CLUSTER_NAMES = {
    0: "🌟 Elite Superstars (High Wage & Skill)",
    1: "💎 Young Prospects (High Potential)",
    2: "🛡️ Reliable Veterans (Solid & Experienced)"
}

# ==========================================
# 2. SIDEBAR - STUDENT INFO & NAVIGATION
# ==========================================
st.sidebar.header("Project Details")
st.sidebar.text("Name: Kyaw Toe Toe Han")
st.sidebar.text("Project: FIFA Player Clustering")
st.sidebar.divider()

# Navigation
menu = st.sidebar.radio(
    "Select Tool:",
    ["📝 AI Scout Report", "📊 Data Analysis"]
)

# ==========================================
# 3. LOAD DATA & MODELS
# ==========================================
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    kmeans_model, scaler = load_models()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run the notebook to generate 'kmeans_model.pkl' and 'scaler.pkl'.")
    st.stop()

# Helper function to load raw data for analysis (Optional, if you have the CSV)
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv('players_22.csv')
    except:
        return None

# ==========================================
# 4. PAGE: AI SCOUT REPORT (Prediction)
# ==========================================
if menu == "📝 AI Scout Report":
    st.title("⚽ AI Football Scout Report")
    st.markdown("Enter player statistics below to categorize them into a scouting cluster.")

    with st.form("player_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_overall = st.number_input("Overall Rating", 1, 99, 75)
            p_potential = st.number_input("Potential Rating", 1, 99, 80)
            p_age = st.number_input("Age", 15, 45, 22)
        with c2:
            p_wage = st.number_input("Wage (EUR)", 0, 1000000, 5000)
            p_value = st.number_input("Value (EUR)", 0, 200000000, 1000000)
            st.caption("*(Growth Potential is calculated automatically)*")
        with c3:
            p_pass = st.number_input("Passing", 1, 99, 60)
            p_shoot = st.number_input("Shooting", 1, 99, 60)
            p_dribble = st.number_input("Dribbling", 1, 99, 60)

        submitted = st.form_submit_button("Generate Report")

        if submitted:
            # 1. Calculate Growth manually
            p_growth = p_potential - p_overall

            # 2. Prepare Data (Must match the exact column order of the scaler)
            input_cols = [
                'overall', 'potential', 'wage_eur', 'value_eur', 
                'age', 'passing', 'shooting', 'dribbling', 'growth_potential'
            ]
            
            input_data = pd.DataFrame([[\n                p_overall, p_potential, p_wage, p_value, 
                p_age, p_pass, p_shoot, p_dribble, p_growth
            ]], columns=input_cols)
            
            # 3. Predict
            try:
                input_scaled = scaler.transform(input_data)
                pred_id = kmeans_model.predict(input_scaled)[0]
                pred_name = CLUSTER_NAMES[pred_id]
                
                st.success(f"✅ Analysis Result: This player is a **{pred_name}**.")
                
                # Visualizing the player stats relative to a generic average
                st.subheader("Player Radar Chart")
                stats_dict = {
                    'Passing': p_pass, 'Shooting': p_shoot, 
                    'Dribbling': p_dribble, 'Overall': p_overall, 'Potential': p_potential
                }
                radar_df = pd.DataFrame(dict(r=list(stats_dict.values()), theta=list(stats_dict.keys())))
                fig = px.line_polar(radar_df, r='r', theta='theta', line_close=True, range_r=[0,100])
                fig.update_traces(fill='toself')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ==========================================
# 5. PAGE: DATA ANALYSIS (New Feature)
# ==========================================
elif menu == "📊 Data Analysis":
    st.title("📊 Project Insights")
    st.info("Here we visualize the relationships used to build the AI model.")

    df = load_raw_data()
    
    if df is not None:
        # Preprocessing for Correlation Matrix
        numeric_features = [
            'overall', 'potential', 'wage_eur', 'value_eur',
            'age', 'passing', 'shooting', 'dribbling'
        ]
        
        # 1. Correlation Matrix
        st.subheader("1. Feature Correlation Matrix")
        st.write("This heatmap shows how different player stats correlate (e.g., Value vs. Wage).")
        
        if set(numeric_features).issubset(df.columns):
            corr = df[numeric_features].corr()
            fig_corr, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig_corr)
        else:
            st.warning("Could not load numeric features for correlation.")

        # 2. Distribution Plot
        st.subheader("2. Player Age vs. Overall Rating")
        fig_scatter = px.scatter(df, x='age', y='overall', color='potential', 
                                 title="Age vs Overall (Colored by Potential)", opacity=0.6)
        st.plotly_chart(fig_scatter)
    else:
        st.warning("⚠️ 'players_22.csv' not found. Please upload the dataset to view the analysis charts.")
