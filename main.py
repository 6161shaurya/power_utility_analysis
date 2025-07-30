# main.py

import os
import pandas as pd

from data_generator import generate_synthetic_outage_data, generate_synthetic_weather_data, generate_synthetic_consumption_data
from power_outage_prediction import preprocess_outage_data, perform_eda_outages, visualize_outage_map, train_outage_prediction_model, forecast_outage_frequency
from power_theft_detection import preprocess_consumption_data, train_theft_detection_model_supervised, detect_theft_unsupervised, perform_eda_consumption

def run_project():
    """
    Main function to run the entire power utility analytics project workflow.
    This includes data generation, preprocessing, EDA, model training, and
    saving outputs.
    """
    print("--- Starting Power Utility Analytics Project ---")

    # --- 1. Data Generation / Loading ---
    print("\n--- Phase 1: Data Generation / Loading ---")
    
    data_dir = 'data'
    output_dir = 'output' 
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True) 
    
    outages_file = os.path.join(data_dir, 'raw_outages.csv')
    weather_file = os.path.join(data_dir, 'raw_weather.csv')
    consumption_file = os.path.join(data_dir, 'raw_consumption.csv')

    if not all(os.path.exists(f) for f in [outages_file, weather_file, consumption_file]):
        print("Raw data files not found. Generating new synthetic data...")
        
        outages_df = generate_synthetic_outage_data(start_date='2023-01-01', end_date='2024-12-31')
        outages_df.to_csv(outages_file, index=False)

        weather_df = generate_synthetic_weather_data(start_date='2023-01-01', end_date='2024-12-31')
        weather_df.to_csv(weather_file, index=False)

        # Generate consumption data with reduced parameters to avoid memory errors
        consumption_df = generate_synthetic_consumption_data(num_consumers=100, end_date='2023-03-31')
        consumption_df.to_csv(consumption_file, index=False)
        
        print("Synthetic data generation complete and saved to 'data/' directory.")
    else:
        print("Raw data files found. Loading existing synthetic data from 'data/' directory.")
        outages_df = pd.read_csv(outages_file)
        weather_df = pd.read_csv(weather_file)
        consumption_df = pd.read_csv(consumption_file)

    # --- 2. Power Outage Prediction Section ---
    print("\n--- Phase 2: Power Outage Prediction Module ---")
    
    print("\n--- Preprocessing Outage Data ---")
    processed_outages_df = preprocess_outage_data(outages_df.copy(), weather_df.copy())

    print("\n--- Performing Outage EDA ---")
    perform_eda_outages(processed_outages_df.copy())

    print("\n--- Generating Outage Map ---")
    visualize_outage_map(processed_outages_df.copy()) 

    print("\n--- Training Outage Prediction Model ---")
    outage_model, daily_df_with_preds, X_test_idx = train_outage_prediction_model(processed_outages_df.copy())

    print("\n--- Forecasting Outage Frequency ---")
    outage_prophet_model, forecast = forecast_outage_frequency(processed_outages_df.copy())

    # --- 3. Power Theft Detection Section ---
    print("\n--- Phase 3: Power Theft Detection Module ---")
    
    print("\n--- Preprocessing Consumption Data ---")
    # --- FIX START: Capture 'le' from preprocessing ---
    processed_consumption_df, consumption_label_encoder = preprocess_consumption_data(consumption_df.copy())
    # --- FIX END ---

    print("\n--- Performing Consumption EDA ---")
    perform_eda_consumption(processed_consumption_df.copy())

    print("\n--- Training Supervised Theft Detection Model ---")
    # --- FIX START: Pass 'le' to the training function ---
    supervised_theft_model, supervised_scaler, supervised_le = \
        train_theft_detection_model_supervised(processed_consumption_df.copy(), le_encoder=consumption_label_encoder)
    # --- FIX END ---

    print("\n--- Running Unsupervised Theft Detection ---")
    df_with_anomalies, iso_forest_model, iso_scaler = \
        detect_theft_unsupervised(processed_consumption_df.copy())

    print("\n--- Project Execution Complete ---")
    print("All data processing, EDA, and model training steps have been run.")
    print("\nNow, run the interactive Streamlit dashboard to visualize results:")
    print("In your terminal, navigate to the project directory and run: ")
    print("`streamlit run dashboard_app.py`")

if __name__ == '__main__':
    run_project()