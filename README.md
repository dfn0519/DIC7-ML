live demo:https://dic7-ml-nmresxlbworci3vysbuxtd.streamlit.app/

# CRISP-DM Linear Regression Streamlit App

This repository contains a single-file Streamlit application (`app.py`) that demonstrates the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) workflow using a synthetic linear regression dataset.

## Features

The application is intuitively structured into the 6 standard phases of the CRISP-DM methodology:

1. **Business Understanding:** Defines the objective of the linear regression task.
2. **Data Understanding:** Displays raw synthetic data, summary statistics, and scatter plots.
3. **Data Preparation:** Handles train-test splitting (80/20) and feature scaling using `StandardScaler`.
4. **Modeling:** Trains a `LinearRegression` model using Scikit-Learn and compares learned parameters against the true synthesized parameters.
5. **Evaluation:** Computes and displays MSE, RMSE, and R² scores, along with an interactive regression line plot against test data.
6. **Deployment:** Provides an interactive prediction input and allows downloading the trained model and scaler as a `.joblib` artifact.

## Local Installation & Usage

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Execute the following command in your terminal:
```bash
streamlit run app.py
```
This will start the local Streamlit server. Open the provided Local URL (usually `http://localhost:8501`) in your web browser.

## Technologies Used
- **Streamlit**: For the interactive web interface.
- **Scikit-learn**: For machine learning data preprocessing, model training, and evaluation.
- **Pandas & NumPy**: For data manipulation and generation.
- **Matplotlib**: For data visualization.
