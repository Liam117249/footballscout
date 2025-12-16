import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import numpy as np

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="FIFA Scout Pro", layout="wide")

st.title("⚽ FIFA Scout Pro: The 'Moneyball' Tool")
st.markdown("""
**Project Goal:** Use Artificial Intelligence to identify player types and find undervalued talent.
""")

# ==========================================
# 2. LOAD DATA & MODELS
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

# ==========================================
# 3. RUN MODEL (Background)
# ==========================================
features_numeric = [
    'overall', 'potential', 'wage_eur', 'value_eur', 
    'age', 'passing', 'shooting', 'dribbling', 'growth_potential'
]

X = df[features_numeric]
X_scaled = scaler.transform(X)
df['Cluster'] = kmeans_model.predict(X_scaled)

# ==========================================
# 4. SIDEBAR - CLUSTER NAMING
# ==========================================
st.sidebar.header("⚙️ Customize AI Groups")
st.sidebar.info("Rename the groups based on what you see in the charts.")

default_names = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4"
}

cluster_names = {}
for i in range(5):
    cluster_names[i] = st.sidebar.text_input(f"Name for Cluster {i}:", default_names[i])

df['Cluster Name'] = df['Cluster'].map(cluster_names)

# ==========================================
# 5. FEATURE 1: MARKET OVERVIEW
# ==========================================
st.header("1. Market Overview 📈")
st.write("See how different player types compare in Skill vs. Wages.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x='overall', y='wage_eur', hue='Cluster Name', data=df, palette='viridis', s=50, alpha=0.6, ax=ax)
plt.xlabel("Overall Rating")
plt.ylabel("Wage (€)")
plt.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig)

# ==========================================
# 6. FEATURE 2: CLUB STRATEGY SCANNER
# ==========================================
st.divider()
st.header("2. Club Strategy Scanner 🛡️")
st.write("Select a club to see their 'DNA' (Player Composition).")

club_list = df['club_name'].sort_values().unique()
selected_club = st.selectbox("Select a Club:", club_list, index=list(club_list).index("Manchester City") if "Manchester City" in list(club_list) else 0)

club_data = df[df['club_name'] == selected_club]
cluster_counts = club_data['Cluster Name'].value_counts().reset_index()
cluster_counts.columns = ['Player Type', 'Count']

fig_pie = px.pie(cluster_counts, values='Count', names='Player Type', title=f"Composition of {selected_club}")
st.plotly_chart(fig_pie)

# ==========================================
# 7. FEATURE 3: THE SMART RECRUITER
# ==========================================
st.divider()
st.header("3. The Smart Recruiter 🕵️‍♂️")
st.write("Find the best players from a specific group that fit your budget.")

col1, col2 = st.columns(2)
with col1:
    target_cluster_name = st.selectbox("Which Player Type do you need?", list(cluster_names.values()))
with col2:
    max_wage = st.slider("Maximum Wage Budget (€)?", min_value=1000, max_value=500000, value=20000, step=1000)

target_cluster_id = [k for k, v in cluster_names.items() if v == target_cluster_name][0]

filtered_players = df[
    (df['Cluster'] == target_cluster_id) & 
    (df['wage_eur'] <= max_wage)
].sort_values(by='overall', ascending=False).head(10)

if not filtered_players.empty:
    st.table(filtered_players[['short_name', 'age', 'overall', 'potential', 'wage_eur', 'club_name']])
else:
    st.warning("No players found! Try increasing your budget.")

# ==========================================
# 8. FEATURE 4: AI SCOUT REPORT (New!)
# ==========================================
st.divider()
st.header("4. AI Scout Report 📝")
st.write("Enter the raw stats of a new player, and the AI will decide their Player Type.")

c1, c2, c3 = st.columns(3)
with c1:
    p_overall = st.number_input("Overall Rating", 40, 99, 70)
    p_potential = st.number_input("Potential Rating", 40, 99, 75)
    p_age = st.number_input("Age", 16, 40, 21)
with c2:
    p_wage = st.number_input("Weekly Wage (€)", 500, 1000000, 5000)
    p_value = st.number_input("Market Value (€)", 0, 200000000, 2000000)
    # Auto-calculate growth
    p_growth = p_potential - p_overall
    st.metric("Growth Potential", p_growth)
with c3:
    p_pass = st.number_input("Passing", 1, 99, 60)
    p_shoot = st.number_input("Shooting", 1, 99, 60)
    p_dribble = st.number_input("Dribbling", 1, 99, 60)

if st.button("Generate Scout Report"):
    # 1. Structure the input data exactly like the training data
    input_data = pd.DataFrame([[
        p_overall, p_potential, p_wage, p_value, 
        p_age, p_pass, p_shoot, p_dribble, p_growth
    ]], columns=features_numeric)
    
    # 2. Scale the data (Crucial step!)
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict the Cluster
    prediction_id = kmeans_model.predict(input_scaled)[0]
    predicted_name = cluster_names[prediction_id]
    
    # 4. Display Result
    st.success(f"✅ Analysis Complete! This player belongs to: **{predicted_name}**")
    
    # Optional: Explanation
    st.info(f"The AI grouped this player into **{predicted_name}** because their stats (Age: {p_age}, Wage: €{p_wage}) match the pattern of other players in this category.")
