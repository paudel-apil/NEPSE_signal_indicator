import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
from reqd_module import calculate_technical_indicators, prepare_lstm_data, build_and_train_lstm_model, predict_lstm_scores, calculate_candle_scores, combine_scores_and_predict

# Set page configuration
st.set_page_config(layout="wide", page_title="NEPSE Trading Signal Predictor")

# Title
st.title("NEPSE Trading Signal Predictor")

# Sidebar for input
st.sidebar.header("Select Dataset")
dataset_dir = "cleaned_dataset"
available_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
selected_file = st.sidebar.selectbox("Select a dataset", available_files, index=0 if available_files else None)

if selected_file and os.path.exists(os.path.join(dataset_dir, selected_file)):
    # Load data
    df = pd.read_csv(os.path.join(dataset_dir, selected_file))
    st.sidebar.success(f"Loaded dataset: {selected_file}")

    # Calculate technical indicators
    with st.spinner("Calculating technical indicators..."):
        df = calculate_technical_indicators(df)

    # Prepare and train LSTM model
    with st.spinner("Preparing and training LSTM model..."):
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)
        lstm_model = build_and_train_lstm_model(X_train, y_train, X_test, y_test)

    # Calculate candle scores
    with st.spinner("Calculating candle scores..."):
        df = calculate_candle_scores(df)

    # Predict LSTM scores
    with st.spinner("Predicting LSTM scores..."):
        deep_scores = predict_lstm_scores(df, lstm_model)

    # Combine scores and get final prediction
    with st.spinner("Combining scores and generating final signal..."):
        final_signal = combine_scores_and_predict(df, deep_scores)

    # Display DataFrame
    st.subheader("Processed Data (Last 10 Rows)")
    st.write(df.tail(10))

    # Visualization 1: Scatter Plot
    st.subheader("LSTM Predictions vs. True Values")
    preds = lstm_model.predict(X_test)
    preds = preds.ravel()
    y_test_original = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 5))]))[:, 0]
    preds_original = scaler.inverse_transform(np.hstack([preds.reshape(-1, 1), np.zeros((len(preds), 5))]))[:, 0]

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.scatter(y_test_original, preds_original, alpha=0.6, label="Predictions vs. True")
    ax1.plot(y_test_original, y_test_original, color="red", linestyle="dashed", label="Ideal Fit")
    ax1.set_xlabel("True Close Price")
    ax1.set_ylabel("Predicted Close Price")
    ax1.set_title("LSTM Predictions vs. True Values")
    ax1.legend()
    st.pyplot(fig1)

    # Visualization 2: Line Plot
    st.subheader("Predicted vs Actual Close Price")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_test_original, label='Actual Close Price')
    ax2.plot(preds_original, label='Predicted Close Price')
    ax2.set_title("Predicted vs Actual Close Price")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Final Prediction
    st.subheader("Final Trading Signal")
    st.success(f"Prediction (as of 11:30 AM +0545, Friday, July 18, 2025): {final_signal}")
else:
    st.sidebar.warning("No valid dataset selected or directory issue. Please ensure 'cleaned_dataset' contains CSV files.")
    st.write("### Instructions")
    st.write("""
    1. Ensure the 'cleaned_dataset' subfolder contains CSV files with the following columns:
       - `close_price`
       - `open_price`
       - `high_price`
       - `low_price`
       - `volume`
       - `turnover`
    2. Select a dataset from the dropdown in the sidebar.
    3. The app will process the data, train the LSTM model, and display the results.
    """)

# Run with: streamlit run main.py