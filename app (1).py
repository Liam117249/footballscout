import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="FIFA 22 Player Clustering", layout="wide")

st.title("⚽ FIFA 22 Player Clustering Tool")
st.markdown("""
This app uses **Machine Learning (K-Means)** to group football players based on their skills and market value.
**Project Goal:** Help scouts identify player types (e.g., "Young Talents" vs "Elite Veterans") automatically.
""")

# ==========================================
# 2. DATA & MODEL LOADING
# ==========================================
@st.cache_data
def load_data():
    """Loads and cleans the FIFA 22 dataset."""
    # Load the CSV file
    try:
        df = pd.read_csv('players_22.csv')
    except FileNotFoundError:
        st.error("⚠️ Error: 'players_22.csv' not found. Please upload it to Colab files.")
        st.stop()

    # Feature Engineering: Re-create 'growth_potential' if it doesn't exist
    if 'growth_potential' not in df.columns:
        df['growth_potential'] = df['potential'] - df['overall']

    # Select the exact same features used in the Notebook
    selected_features = [
        'short_name', 'overall', 'potential', 'wage_eur', 
        'value_eur', 'age', 'passing', 'shooting', 
        'dribbling', 'growth_potential'
    ]
    
    # Drop rows with missing values to prevent errors
    df_clean = df[selected_features].dropna()
    return df_clean

@st.cache_resource
def load_model():
    """Loads the saved K-Means model and Scaler."""
    try:
        with open('kmeans_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found. Please run your Notebook code to generate 'kmeans_model.pkl' and 'scaler.pkl'.")
        st.stop()

# Execute Loading
df = load_data()
kmeans_model, scaler = load_model()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🔍 Options")
show_raw = st.sidebar.checkbox("Show Raw Data", False)
st.sidebar.info("Use the tools below to analyze the clusters.")

# ==========================================
# 4. MAIN VISUALIZATION
# ==========================================

# Prepare data for prediction/plotting (Remove Name column)
features_numeric = [
    'overall', 'potential', 'wage_eur', 'value_eur', 
    'age', 'passing', 'shooting', 'dribbling', 'growth_potential'
]

# Scale the data using the loaded scaler
X = df[features_numeric]
X_scaled = scaler.transform(X)

# Assign Clusters to the DataFrame
df['Cluster'] = kmeans_model.predict(X_scaled)

# -- SECTION A: Raw Data --
if show_raw:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50))

# -- SECTION B: PCA Visualization (The "Mastery" Chart) --
st.divider()
st.subheader("📊 Cluster Visualization (PCA)")
st.write("We use **Principal Component Analysis (PCA)** to reduce 7+ dimensions into a 2D Scatter Plot.")

# run PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    data=df, 
    palette='viridis', 
    alpha=0.6, 
    ax=ax
)
plt.title("FIFA 22 Player Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
st.pyplot(fig)

# -- SECTION C: Cluster Interpretation --
st.subheader("📝 Cluster Interpretation")
st.write("Below are the average statistics for each cluster. Use this to name your groups (e.g., 'Cluster 0 = Young Stars').")

# FIX: Added numeric_only=True to prevent string errors
cluster_summary = df.groupby('Cluster')[features_numeric].mean(numeric_only=True)
st.dataframe(cluster_summary)

# ==========================================
# 5. PREDICTION TOOL (Interactive GUI)
# ==========================================
st.divider()
st.subheader("🤖 Scout a New Player")
st.markdown("Enter a player's stats to see which cluster they belong to.")

col1, col2, col3 = st.columns(3)

with col1:
    p_overall = st.number_input("Overall Rating", 40, 99, 70)
    p_potential = st.number_input("Potential Rating", 40, 99, 75)
    p_age = st.number_input("Age", 16, 45, 21)

with col2:
    p_wage = st.number_input("Wage (€)", 500, 1000000, 5000)
    p_value = st.number_input("Value (€)", 0, 200000000, 2000000)
    # Auto-calculate growth
    p_growth = p_potential - p_overall
    st.metric("Growth Potential", p_growth)

with col3:
    p_pass = st.number_input("Passing", 1, 99, 60)
    p_shoot = st.number_input("Shooting", 1, 99, 60)
    p_dribble = st.number_input("Dribbling", 1, 99, 60)

# Predict Button
if st.button("Predict Cluster"):
    # 1. Organize input into a DataFrame
    input_data = pd.DataFrame([[
        p_overall, p_potential, p_wage, p_value, 
        p_age, p_pass, p_shoot, p_dribble, p_growth
    ]], columns=features_numeric)
    
    # 2. Scale the input using the SAME scaler as the model
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = kmeans_model.predict(input_scaled)[0]
    
    st.success(f"✅ This player belongs to **Cluster {prediction}**")
