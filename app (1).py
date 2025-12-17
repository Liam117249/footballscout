import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# ==========================================
# 1. APP CONFIGURATION & CLUSTER NAMES
# ==========================================
st.set_page_config(page_title="FIFA Scout Pro", layout="wide")

# Hardcoded names based on your Cluster Analysis
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
st.sidebar.text("Student ID: [ENTER ID HERE]")  # <--- Type your ID here
st.sidebar.text("Class: [ENTER CLASS HERE]")     # <--- Type your Class here
st.sidebar.text("Project: FIFA Player Clustering")
st.sidebar.text("Professor: Tr. NN")

st.sidebar.divider()

# Navigation
st.sidebar.header("Choose a Feature")
menu = st.sidebar.radio(
    "Select Tool:",
    ["🛡️ Club Strategy Scanner", "💰 The Smart Recruiter", "📝 AI Scout Report"]
)

# ==========================================
# 3. LOAD DATA & MODELS
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('players_22.csv')
    except FileNotFoundError:
        st.error("⚠️ Error: 'players_22.csv' not found.")
        st.stop()

    if 'growth_potential' not in df.columns:
        df['growth_potential'] = df['potential'] - df['overall']

    selected_features = [
        'short_name', 'club_name', 'overall', 'potential', 'wage_eur', 
        'value_eur', 'age', 'passing', 'shooting', 
        'dribbling', 'growth_potential'
    ]
    df_clean = df[selected_features].dropna()
    return df_clean

@st.cache_resource
def load_model():
    try:
        with open('kmeans_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found.")
        st.stop()

df = load_data()
kmeans_model, scaler = load_model()

# Run Predictions in Background
features_numeric = [
    'overall', 'potential', 'wage_eur', 'value_eur', 
    'age', 'passing', 'shooting', 'dribbling', 'growth_potential'
]
X = df[features_numeric]
X_scaled = scaler.transform(X)
df['Cluster'] = kmeans_model.predict(X_scaled)
df['Cluster Name'] = df['Cluster'].map(CLUSTER_NAMES)

# ==========================================
# 4. MAIN FEATURES
# ==========================================

# --- FEATURE 1: CLUB STRATEGY SCANNER ---
if menu == "🛡️ Club Strategy Scanner":
    st.header("🛡️ Club Strategy Scanner")
    st.write("Analyze a football club's 'DNA' to see what kind of players they recruit.")
    
    # Select Club
    club_list = df['club_name'].sort_values().unique()
    selected_club = st.selectbox("Select a Club:", club_list)
    
    # Filter Data
    club_data = df[df['club_name'] == selected_club]
    cluster_counts = club_data['Cluster Name'].value_counts().reset_index()
    cluster_counts.columns = ['Player Type', 'Count']
    
    # Display Pie Chart
    fig_pie = px.pie(
        cluster_counts, 
        values='Count', 
        names='Player Type', 
        title=f"Player Composition: {selected_club}",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_pie)

# --- FEATURE 2: THE SMART RECRUITER ---
elif menu == "💰 The Smart Recruiter":
    st.header("💰 The Smart Recruiter")
    st.write("Find the best players that fit your budget and desired playing style.")
    
    col1, col2 = st.columns(2)
    with col1:
        target_cluster = st.selectbox("1. I want this Player Type:", list(CLUSTER_NAMES.values()))
    with col2:
        max_wage = st.slider("2. My Weekly Budget is (€):", 1000, 300000, 20000, step=1000)
    
    # Find Cluster ID from Name
    target_id = [k for k, v in CLUSTER_NAMES.items() if v == target_cluster][0]
    
    # Filter Results
    results = df[
        (df['Cluster'] == target_id) & 
        (df['wage_eur'] <= max_wage)
    ].sort_values(by='overall', ascending=False).head(10)
    
    st.subheader("Top Recommendations")
    if not results.empty:
        st.table(results[['short_name', 'age', 'overall', 'wage_eur', 'club_name']])
    else:
        st.warning("No players found. Try increasing your budget.")
# --- FEATURE 3: AI SCOUT REPORT (FIXED) ---
elif menu == "📝 AI Scout Report":
    st.header("📝 AI Scout Report")
    st.write("Enter a player's raw statistics to identify which category they belong to.")

    # We use a 'form' so the app doesn't reload until you press Submit
    with st.form("scout_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_overall = st.number_input("Overall Rating", 40, 99, 70)
            p_potential = st.number_input("Potential Rating", 40, 99, 75)
            p_age = st.number_input("Age", 16, 40, 21)
        with c2:
            p_wage = st.number_input("Weekly Wage (€)", 500, 1000000, 5000)
            p_value = st.number_input("Market Value (€)", 0, 200000000, 2000000)
            st.caption("*(Growth Potential is calculated automatically)*")
        with c3:
            p_pass = st.number_input("Passing", 1, 99, 60)
            p_shoot = st.number_input("Shooting", 1, 99, 60)
            p_dribble = st.number_input("Dribbling", 1, 99, 60)

        # The Button is now inside the form
        submitted = st.form_submit_button("Generate Report")

        if submitted:
            # 1. Calculate Growth manually here
            p_growth = p_potential - p_overall

            # 2. Prepare Data (Must match the exact column order of the scaler)
            input_cols = [
                'overall', 'potential', 'wage_eur', 'value_eur', 
                'age', 'passing', 'shooting', 'dribbling', 'growth_potential'
            ]
            
            input_data = pd.DataFrame([[
                p_overall, p_potential, p_wage, p_value, 
                p_age, p_pass, p_shoot, p_dribble, p_growth
            ]], columns=input_cols)
            
            # 3. Predict
            try:
                input_scaled = scaler.transform(input_data)
                pred_id = kmeans_model.predict(input_scaled)[0]
                pred_name = CLUSTER_NAMES[pred_id]
                
                st.success(f"✅ Analysis Result: This player is a **{pred_name}**.")
                st.info(f"Why? Their stats (Age: {p_age}, Overall: {p_overall}) match the patterns of the **{pred_name}** group.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
