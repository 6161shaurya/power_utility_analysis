# data_generator.py

import pandas as pd
import numpy as np
from datetime import timedelta
import os # Used for checking/creating directories

def generate_synthetic_outage_data(start_date='2023-01-01', end_date='2024-12-31', num_outages=5000):
    """
    Generates synthetic power outage data for a specified period.
    
    Args:
        start_date (str): The start date for generating outage data (YYYY-MM-DD).
        end_date (str): The end date for generating outage data (YYYY-MM-DD).
        num_outages (int): The total number of synthetic outage events to generate.
        
    Returns:
        pd.DataFrame: A DataFrame containing generated outage records with StartTime,
                      EndTime, DurationMinutes, Feeder, Cause, and AffectedCustomers.
    """
    np.random.seed(42) # For reproducibility of random data

    # Define a time range spanning from start_date to end_date, with hourly frequency
    date_range_outage = pd.date_range(start=start_date, end=end_date, freq='h') # Corrected 'H' to 'h'

    # Randomly select 'num_outages' start times from the defined date range
    outage_start_times = pd.Series(date_range_outage).sample(n=num_outages, replace=True).sort_values().reset_index(drop=True)

    # Generate random durations for each outage (between 15 minutes and 8 hours)
    outage_durations_minutes = np.random.randint(15, 8*60, num_outages)
    
    # Define a list of hypothetical feeders (e.g., 10 different geographical feeders)
    feeders = [f'Feeder_{i}' for i in range(1, 11)]
    # Randomly assign a feeder to each outage event
    outage_feeders = np.random.choice(feeders, num_outages)

    # Define common causes of power outages and their probabilities
    causes = ['Equipment Failure', 'Overload', 'Tree Contact', 'Weather (Heavy Rain)',
              'Weather (High Wind)', 'Planned Maintenance', 'Unknown']
    # Randomly assign a cause to each outage based on predefined probabilities
    outage_causes_array = np.random.choice(causes, num_outages, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.2])

    # Initialize affected_customers count for each outage
    affected_customers = np.random.randint(10, 500, num_outages) # Base range for affected customers
    
    # --- FIX for AttributeError: 'numpy.ndarray' object has no attribute 'str' ---
    # Convert the numpy array of causes to a Pandas Series to use .str accessor for string operations
    outage_causes_series = pd.Series(outage_causes_array)
    
    # Apply a bias: Overload causes typically affect more customers
    affected_customers[outage_causes_series == 'Overload'] = \
        np.random.randint(200, 1000, sum(outage_causes_series == 'Overload'))
    
    # Apply a bias: Weather-related causes typically affect even more customers
    affected_customers[outage_causes_series.str.startswith('Weather')] = \
        np.random.randint(300, 1500, sum(outage_causes_series.str.startswith('Weather')))
    # --- END FIX ---

    # Create the DataFrame from the generated data
    df = pd.DataFrame({
        'OutageID': range(1, num_outages + 1), # Unique identifier for each outage
        'StartTime': outage_start_times,
        'DurationMinutes': outage_durations_minutes,
        'Feeder': outage_feeders,
        'Cause': outage_causes_array, # Use the original numpy array here
        'AffectedCustomers': affected_customers
    })
    
    # Calculate the EndTime for each outage based on StartTime and DurationMinutes
    df['EndTime'] = df['StartTime'] + pd.to_timedelta(df['DurationMinutes'], unit='m')
    
    # Ensure EndTime is always strictly after StartTime. This handles edge cases
    # where tiny random durations might result in EndTime <= StartTime.
    df.loc[df['EndTime'] <= df['StartTime'], 'EndTime'] = df['StartTime'] + pd.to_timedelta(15, unit='m')

    print(f"Generated {len(df)} synthetic outage records.")
    return df

def generate_synthetic_weather_data(start_date='2023-01-01', end_date='2024-12-31'):
    """
    Generates synthetic hourly weather data for Rampur over a specified period.
    Includes seasonal temperature/humidity, sporadic rainfall (higher in monsoon),
    wind speed (with occasional high gusts), and cloud cover.
    
    Args:
        start_date (str): The start date for weather data (YYYY-MM-DD).
        end_date (str): The end date for weather data (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: A DataFrame containing hourly weather records with DateTime,
                      Temperature_C, Humidity_%, Rainfall_mm, WindSpeed_mps, CloudCover_%.
    """
    np.random.seed(42) # For reproducibility

    # Define the time range for hourly weather data
    date_range_weather = pd.date_range(start=start_date, end=end_date, freq='h') # Corrected 'H' to 'h'
    df = pd.DataFrame({'DateTime': date_range_weather})
    
    num_hours = len(df)
    days = np.arange(num_hours) # Array representing hours from start for sinusoidal patterns
    annual_cycle = 365.25 * 24 # Total hours in a year for seasonality calculations

    # Simulate Temperature (in Celsius): Base temperature with seasonal variation and noise
    temp_base = 25 # Average temperature in Rampur (arbitrary)
    temp_amplitude = 15 # Max deviation from average (e.g., from 10C to 40C)
    # Sinusoidal pattern peaks in summer (e.g., around June-July) due to +np.pi/2 phase shift
    temp_seasonal = temp_base + temp_amplitude * np.sin(2 * np.pi * days / annual_cycle + np.pi/2)
    df['Temperature_C'] = temp_seasonal + np.random.normal(0, 3, num_hours) # Add random normal noise

    # Simulate Humidity (%): Generally inverse to temperature, with noise. Clipped to realistic range.
    df['Humidity_%'] = np.clip(80 - temp_amplitude * np.sin(2 * np.pi * days / annual_cycle + np.pi/2) + np.random.normal(0, 5, num_hours), 30, 95)

    # Simulate Rainfall (mm): Mostly low, but higher in monsoon season
    df['Rainfall_mm'] = np.random.rand(num_hours) * 2 # Baseline light rain
    
    # --- FIX for OutOfBoundsDatetime error ---
    # Define monsoon months directly using month numbers (June=6, September=9)
    monsoon_months = [6, 7, 8, 9] 
    
    # Create a boolean mask for dates falling within the monsoon months
    is_monsoon = df['DateTime'].dt.month.isin(monsoon_months)
    # --- END FIX ---
    
    # Increase rainfall significantly during monsoon periods
    df.loc[is_monsoon, 'Rainfall_mm'] = np.random.rand(is_monsoon.sum()) * 15 # Higher rain in monsoon

    # Simulate Wind Speed (m/s): General range with occasional high wind events
    df['WindSpeed_mps'] = np.random.uniform(0, 10, num_hours) # Typical wind speeds
    # Introduce specific high wind events randomly
    high_wind_event_indices = np.random.choice(num_hours, 50, replace=False) # 50 random hours of high wind
    df.loc[high_wind_event_indices, 'WindSpeed_mps'] = np.random.uniform(15, 25, len(high_wind_event_indices)) # Stronger winds

    # Simulate Cloud Cover (%): Random fluctuations between 0% and 100%
    df['CloudCover_%'] = np.random.uniform(0, 100, num_hours)

    print(f"Generated {len(df)} synthetic weather records.")
    return df

def generate_synthetic_consumption_data(start_date='2023-01-01', end_date='2024-03-31', num_consumers=100):
    """
    Generates synthetic hourly electricity consumption data for multiple consumers.
    Simulates different connection types (Residential, Commercial, Industrial),
    their typical load profiles, and introduces periods of 'theft' for some consumers.
    
    Args:
        start_date (str): The start date for consumption data (YYYY-MM-DD).
        end_date (str): The end date for consumption data (YYYY-MM-DD).
        num_consumers (int): The number of synthetic consumers to generate data for.
                              **Reduced default to 100 for memory management.**
        
    Returns:
        pd.DataFrame: A DataFrame containing hourly consumption records per consumer,
                      with DateTime, Connection_Type, SanctionedLoad_kW, Consumption_kWh,
                      IsTheft (label), and TheftAmount_kWh.
    """
    np.random.seed(42) # For reproducibility

    # Define the hourly time range for consumption data
    date_range_consumption = pd.date_range(start=start_date, end=end_date, freq='h') # Corrected 'H' to 'h'
    num_hours = len(date_range_consumption)

    # Generate unique Consumer IDs
    consumer_ids = [f'C_{i:04d}' for i in range(1, num_consumers + 1)]
    connection_types = ['Residential', 'Commercial', 'Industrial']
    
    # Define base consumption profiles for each connection type
    type_profiles = {
        'Residential': {'base_kwh': 0.1, 'peak_amp': 0.5, 'std': 0.05, 'peak_hours': [19, 22]}, # Evening peak
        'Commercial': {'base_kwh': 0.3, 'peak_amp': 0.7, 'std': 0.1, 'peak_hours': [9, 17]},  # Day peak
        'Industrial': {'base_kwh': 0.5, 'peak_amp': 0.2, 'std': 0.15, 'peak_hours': [0, 23]} # More consistent, higher base
    }

    data = [] # List to store all generated consumption records
    
    # Select a percentage of consumers to be 'theft' cases for supervised learning
    theft_consumers_ids = np.random.choice(consumer_ids, int(num_consumers * 0.05), replace=False) # e.g., 5% of consumers are theft

    for consumer_id in consumer_ids:
        # Randomly assign a connection type to each consumer based on typical distribution
        conn_type = np.random.choice(connection_types, p=[0.7, 0.2, 0.1]) # 70% Residential, 20% Commercial, 10% Industrial
        profile = type_profiles[conn_type]
        
        # Sanctioned load for the consumer, based on their connection type's profile
        sanctioned_load = np.random.uniform(profile['base_kwh'] * 100, profile['base_kwh'] * 200) 

        is_theft = consumer_id in theft_consumers_ids # Flag if this consumer is a theft case
        theft_start_idx = None
        theft_end_idx = None
        theft_reduction_factor = 0 # Proportion of consumption hidden by theft

        if is_theft:
            # Simulate a continuous period of theft within the data range for each theft consumer
            # --- FIX for ValueError: high <= 0 in np.random.randint ---
            # Ensure theft_duration_hours is never greater than the total available hours.
            # Max possible theft duration for a given num_hours to allow at least 1 possible start index (index 0).
            max_possible_theft_duration = num_hours 
            if max_possible_theft_duration == 0: # Handle case of 0 hours
                max_possible_theft_duration = 1

            # Define min and max theft durations (e.g., 1 month to 6 months)
            min_theft_duration_hours = 30 * 24 # Minimum 1 month in hours
            max_theft_duration_hours_bound = 180 * 24 # Maximum 6 months in hours

            # Cap the random theft duration to ensure it fits within the dataset's length.
            # Also ensure it's at least min_theft_duration_hours.
            actual_max_theft_duration = min(max_theft_duration_hours_bound, max_possible_theft_duration)

            # Ensure the minimum duration is not greater than the maximum allowed duration.
            if min_theft_duration_hours > actual_max_theft_duration:
                # If dataset is too short for a 1-month theft, just set it to actual_max_theft_duration (e.g., 1 hour)
                min_theft_duration_hours = actual_max_theft_duration 
                if min_theft_duration_hours < 1: # Safety for extremely short datasets
                    min_theft_duration_hours = 1
            
            # Randomly select a theft duration within the valid range
            theft_duration_hours = np.random.randint(min_theft_duration_hours, actual_max_theft_duration + 1)
            
            # Calculate the maximum valid starting index for the chosen theft duration
            # `high` argument in randint is exclusive, so add 1 to `max_start_index` to include it.
            max_start_index_for_randint = max(0, num_hours - theft_duration_hours) # Ensure it's not negative
            
            theft_start_idx = np.random.randint(0, max_start_index_for_randint + 1) # Random start hour (inclusive of max_start_index)
            # --- END FIX ---

            theft_end_idx = theft_start_idx + theft_duration_hours
            theft_reduction_factor = np.random.uniform(0.3, 0.8) # 30-80% consumption reduction

        for i, dt in enumerate(date_range_consumption):
            hour = dt.hour
            day_of_year_norm = dt.dayofyear / 365.25 # Normalized day of year for annual seasonality calculation

            # Calculate base consumption for the hour, with some random fluctuation
            base_consumption_kwh = profile['base_kwh'] + np.random.normal(0, profile['std'])
            
            # Apply daily/hourly peak patterns based on connection type
            peak_factor = 1.0
            if profile['peak_hours'][0] <= hour <= profile['peak_hours'][1]:
                # Stronger peak effect around the typical peak hours
                peak_factor = 1.0 + profile['peak_amp'] * (1 - abs(hour - np.mean(profile['peak_hours'])) / ((profile['peak_hours'][1] - profile['peak_hours'][0]) / 2 + 0.1))

            # Apply weekly seasonality (e.g., lower consumption on weekends for commercial users)
            weekday_factor = 1.0
            if conn_type == 'Commercial' and dt.dayofweek >= 5: # Saturday (5) or Sunday (6)
                weekday_factor = 0.7 # Commercial use might drop by 30% on weekends

            # Apply annual seasonality (e.g., higher consumption in summer for AC)
            annual_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year_norm + np.pi/2) # Peaks around summer (June-August)

            # Combine all factors to get the initial simulated consumption
            consumption_kwh = (base_consumption_kwh * peak_factor * weekday_factor * annual_factor) + np.random.normal(0, profile['std']) # Add final noise
            
            # Ensure consumption is never negative
            consumption_kwh = max(0, consumption_kwh)

            # Apply theft effect if the consumer is a theft case and currently in their theft period
            actual_theft_amount = 0
            if is_theft and (theft_start_idx <= i < theft_end_idx):
                actual_theft_amount = consumption_kwh * theft_reduction_factor # Calculate the amount stolen
                consumption_kwh *= (1 - theft_reduction_factor) # Reduce the reported consumption

            # Append the record to the data list
            data.append({
                'ConsumerID': consumer_id,
                'DateTime': dt,
                'Connection_Type': conn_type,
                'SanctionedLoad_kW': sanctioned_load, # Sanctioned load in kW
                'Consumption_kWh': consumption_kwh, # Hourly consumption in kWh
                'IsTheft': is_theft, # True/False label for supervised learning (1 if theft, 0 otherwise)
                'TheftAmount_kWh': actual_theft_amount # The actual amount of kWh stolen (for internal analysis)
            })

    # Create final DataFrame from the collected records
    df = pd.DataFrame(data)
    # Final check to ensure no negative consumption values exist after theft reduction
    df['Consumption_kWh'] = np.maximum(0, df['Consumption_kWh'])
    print(f"Generated {len(df)} synthetic consumption records for {num_consumers} consumers.")
    return df

# This block executes only when data_generator.py is run directly (e.g., python data_generator.py)
# It's primarily for testing the data generation functions independently.
# In the main project workflow, these functions are called from main.py.
if __name__ == '__main__':
    # Define the directory where the generated data will be saved
    data_dir = 'data'
    # Create the directory if it does not already exist
    os.makedirs(data_dir, exist_ok=True)

    print("--- Generating Synthetic Data (for direct execution of data_generator.py) ---")
    
    # Generate and save outage data (full range)
    outage_df = generate_synthetic_outage_data(start_date='2023-01-01', end_date='2024-12-31')
    outage_df.to_csv(os.path.join(data_dir, 'raw_outages.csv'), index=False)
    print(f"Outage data saved to {os.path.join(data_dir, 'raw_outages.csv')}")

    # Generate and save weather data (full range)
    weather_df = generate_synthetic_weather_data(start_date='2023-01-01', end_date='2024-12-31')
    weather_df.to_csv(os.path.join(data_dir, 'raw_weather.csv'), index=False)
    print(f"Weather data saved to {os.path.join(data_dir, 'raw_weather.csv')}")

    # Generate and save consumption data (REDUCED RANGE for direct execution)
    # This ensures that even if data_generator.py is run alone, it's not generating massive files.
    consumption_df = generate_synthetic_consumption_data(num_consumers=100, end_date='2023-03-31')
    consumption_df.to_csv(os.path.join(data_dir, 'raw_consumption.csv'), index=False)
    print(f"Consumption data saved to {os.path.join(data_dir, 'raw_consumption.csv')}")

    print("\nSynthetic data generation complete and saved to 'data/' directory.")