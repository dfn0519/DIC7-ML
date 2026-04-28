import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io

# Optimize page layout
st.set_page_config(page_title="CRISP-DM Linear Regression", layout="wide")

# --- Cache data generation to optimize speed ---
@st.cache_data
def generate_data(n, noise_variance, seed):
    np.random.seed(seed)
    x = np.random.uniform(-100, 100, n)
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-50, 50)
    noise_mean = np.random.uniform(-10, 10)
    noise = np.random.normal(noise_mean, np.sqrt(noise_variance), n)
    y = a * x + b + noise
    
    df = pd.DataFrame({'X': x, 'y': y})
    return df, a, b, noise_mean

# --- Sidebar UI ---
st.sidebar.title("Data Parameters")
n_samples = st.sidebar.slider("Sample Size (n)", min_value=100, max_value=1000, value=500, step=10)
noise_var = st.sidebar.slider("Noise Variance", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
seed = st.sidebar.slider("Random Seed", min_value=0, max_value=1000, value=42, step=1)

generate_btn = st.sidebar.button("Generate Data")

# Initialize session state for data
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

if generate_btn:
    st.session_state.data_generated = True

st.title("CRISP-DM Workflow: Linear Regression")

# 1. Business Understanding
st.header("1. Business Understanding")
st.markdown("""
**Objective:** Predict a continuous target variable (`y`) based on a single continuous feature (`X`).

**Scenario:** We are demonstrating a synthetic linear regression problem to walk through the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology. The true relationship is linear but includes some normally distributed noise.
""")

st.divider()

# Proceed only if data is generated or default generated
if not st.session_state.data_generated:
    st.info("Please click 'Generate Data' in the sidebar to start the workflow.")
else:
    df, true_a, true_b, true_noise_mean = generate_data(n_samples, noise_var, seed)

    # 2. Data Understanding
    st.header("2. Data Understanding")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
    st.subheader("Data Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df['X'], df['y'], alpha=0.5, label='Data Points', color='royalblue')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Scatter Plot of X vs y')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    
    st.divider()

    # 3. Data Preparation
    st.header("3. Data Preparation")
    st.write("Splitting data into training (80%) and testing (20%) sets, and scaling features using `StandardScaler`.")
    
    X = df[['X']]
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.write(f"- **Training set size:** {X_train.shape[0]} samples")
    st.write(f"- **Testing set size:** {X_test.shape[0]} samples")

    st.divider()

    # 4. Modeling
    st.header("4. Modeling")
    st.write("Training a simple `LinearRegression` model using the scaled training data.")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Inverse transform to get original scale coefficients for comparison
    learned_a = model.coef_[0] / scaler.scale_[0]
    learned_b = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("True Parameters (Data Gen)")
        st.write(f"- **Slope (a):** `{true_a:.4f}`")
        st.write(f"- **Intercept (b):** `{true_b:.4f}`")
        st.write(f"- **Noise Mean:** `{true_noise_mean:.4f}`")
    with col4:
        st.subheader("Learned Parameters (Original Scale)")
        st.write(f"- **Slope:** `{learned_a:.4f}`")
        st.write(f"- **Intercept:** `{learned_b:.4f}`")

    st.divider()

    # 5. Evaluation
    st.header("5. Evaluation")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Training Metrics")
        st.metric("MSE", f"{mse_train:.4f}")
        st.metric("RMSE", f"{rmse_train:.4f}")
        st.metric("R² Score", f"{r2_train:.4f}")
    with col6:
        st.subheader("Testing Metrics")
        st.metric("MSE", f"{mse_test:.4f}")
        st.metric("RMSE", f"{rmse_test:.4f}")
        st.metric("R² Score", f"{r2_test:.4f}")

    st.subheader("Regression Line vs Test Data")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(X_test, y_test, alpha=0.5, label='Test Data Points', color='darkorange')
    
    # Plotting regression line
    x_range = np.linspace(df['X'].min(), df['X'].max(), 100)
    x_range_df = pd.DataFrame({'X': x_range})
    x_range_scaled = scaler.transform(x_range_df)
    y_range_pred = model.predict(x_range_scaled)
    
    ax2.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title('Test Data and Regression Line')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

    st.divider()

    # 6. Deployment
    st.header("6. Deployment")
    st.write("Deploy the model to make new predictions and provide a downloadable artifact.")
    
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Make a Prediction")
        user_input = st.number_input("Enter a value for X:", value=0.0, step=1.0)
        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame({'X': [user_input]})
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted y for X = {user_input} is: **{prediction:.4f}**")
            
    with col8:
        st.subheader("Save Model Artifact")
        st.write("Export the trained model and scaler to a `.joblib` file.")
        
        # Save model and scaler together
        export_dict = {
            'model': model,
            'scaler': scaler
        }
        
        # Buffer to save the joblib object
        buffer = io.BytesIO()
        joblib.dump(export_dict, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="Download Model & Scaler (.joblib)",
            data=buffer,
            file_name="linear_regression_model.joblib",
            mime="application/octet-stream"
        )
