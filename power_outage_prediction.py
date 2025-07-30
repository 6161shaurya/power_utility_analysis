# power_outage_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from prophet import Prophet # For time series forecasting

# Set plotting style for better visualization aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7) # Default figure size for plots


def preprocess_outage_data(outages_df, weather_df):
    """
    Cleans and merges outage and weather data, and performs feature engineering.
    This function prepares the data for subsequent analysis and modeling.

    Args:
        outages_df (pd.DataFrame): DataFrame containing raw power outage records.
        weather_df (pd.DataFrame): DataFrame containing raw hourly weather data.

    Returns:
        pd.DataFrame: Merged and preprocessed DataFrame.
    """
    print("Starting outage data preprocessing...")

    # Ensure datetime columns are correctly parsed
    outages_df['StartTime'] = pd.to_datetime(outages_df['StartTime'])
    outages_df['EndTime'] = pd.to_datetime(outages_df['EndTime'])
    weather_df['DateTime'] = pd.to_datetime(weather_df['DateTime'])

    # --- Feature Engineering from Outage Timestamps ---
    outages_df['OutageDay'] = outages_df['StartTime'].dt.date
    outages_df['OutageHour'] = outages_df['StartTime'].dt.hour
    outages_df['OutageDayOfWeek'] = outages_df['StartTime'].dt.dayofweek # Monday=0, Sunday=6
    outages_df['OutageMonth'] = outages_df['StartTime'].dt.month
    outages_df['OutageYear'] = outages_df['StartTime'].dt.year
    outages_df['OutageMinute'] = outages_df['StartTime'].dt.minute

    # Create a rounded timestamp for merging with weather data.
    # We round to the nearest hour to align with hourly weather data.
    outages_df['MergeTime'] = outages_df['StartTime'].dt.round('H')

    # --- Merge Outage Data with Weather Data ---
    # Set 'DateTime' as index for the weather DataFrame to prepare for merging.
    weather_df_hourly = weather_df.set_index('DateTime')
    
    # Use merge_asof for time-series merging.
    # It merges on 'MergeTime' from outages_df to 'DateTime' from weather_df_hourly,
    # finding the nearest weather reading for each outage start time.
    merged_df = pd.merge_asof(
        outages_df.sort_values('MergeTime'), # Left DataFrame must be sorted
        weather_df_hourly.reset_index().sort_values('DateTime'), # Right DataFrame must be sorted
        left_on='MergeTime',
        right_on='DateTime',
        direction='nearest' # Takes the nearest weather reading in time
    )

    # Drop redundant merge columns to keep the DataFrame clean
    merged_df.drop(columns=['DateTime', 'MergeTime'], inplace=True)

    # --- Convert Categorical Features ---
    # Convert 'Cause' and 'Feeder' columns to categorical data types.
    # This is memory-efficient and can be beneficial for some models.
    merged_df['Cause'] = merged_df['Cause'].astype('category')
    merged_df['Feeder'] = merged_df['Feeder'].astype('category')

    print("Outage data preprocessing complete.")
    return merged_df

def perform_eda_outages(df):
    """
    Performs Exploratory Data Analysis (EDA) on the preprocessed outage data.
    Generates various plots to understand outage patterns, causes, and relationships
    with other features like weather.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing outage and weather data.
    """
    print("Performing EDA for power outages...")

    print(f"Total outages: {len(df)}")
    print(f"Time period: {df['StartTime'].min()} to {df['StartTime'].max()}")

    # 1. Outage Frequency Over Time (Monthly Trend)
    plt.figure(figsize=(15, 6))
    # Addressed FutureWarning: 'M' is deprecated, use 'ME' for Month End
    df.set_index('StartTime').resample('ME')['OutageID'].count().plot()
    plt.title('Monthly Power Outage Frequency')
    plt.xlabel('Date')
    plt.ylabel('Number of Outages')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Outage Duration Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['DurationMinutes'], bins=50, kde=True)
    plt.title('Distribution of Outage Durations (Minutes)')
    plt.xlabel('Duration (Minutes)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Outage Causes
    plt.figure(figsize=(12, 7))
    # Order bars by frequency (most common cause at the top)
    sns.countplot(data=df, y='Cause', order=df['Cause'].value_counts().index, palette='viridis')
    plt.title('Distribution of Outage Causes')
    plt.xlabel('Number of Outages')
    plt.ylabel('Cause')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

    # 4. Outages by Feeder (Identify problematic feeders)
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, y='Feeder', order=df['Feeder'].value_counts().index, palette='plasma')
    plt.title('Number of Outages by Feeder')
    plt.xlabel('Number of Outages')
    plt.ylabel('Feeder')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

    # 5. Relationship between Outages and Weather (Example: Temperature by Cause)
    # This helps understand if certain causes are more prevalent at specific temperatures.
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Cause', y='Temperature_C', palette='coolwarm')
    plt.title('Temperature Distribution by Outage Cause')
    plt.xlabel('Outage Cause')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 6. Outages by Hour of Day and Day of Week (Temporal Patterns)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.countplot(data=df, x='OutageHour', palette='crest', ax=axes[0])
    axes[0].set_title('Outage Frequency by Hour of Day')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Number of Outages')
    axes[0].grid(axis='y')

    sns.countplot(data=df, x='OutageDayOfWeek', palette='magma', ax=axes[1])
    axes[1].set_title('Outage Frequency by Day of Week')
    axes[1].set_xlabel('Day of Week (0=Monday, 6=Sunday)')
    axes[1].set_ylabel('Number of Outages')
    axes[1].grid(axis='y')
    
    plt.tight_layout()
    plt.show()

    # 7. Correlation Matrix of Numerical Features
    # Helps identify relationships between numerical features, including weather and outage attributes.
    numerical_cols = ['DurationMinutes', 'AffectedCustomers', 'Temperature_C', 'Humidity_%',
                      'Rainfall_mm', 'WindSpeed_mps', 'CloudCover_%']
    # Filter out columns that might not exist or have no variance (e.g., if only one value)
    numerical_cols = [col for col in numerical_cols if col in df.columns and df[col].nunique() > 1]

    if numerical_cols:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numerical columns with variance for correlation matrix.")

    print("EDA for power outages complete.")

def visualize_outage_map(df, feeder_coords=None, center_coords=[28.837, 79.037]):
    """
    Visualizes outages on an interactive geographical map using Folium.
    Circles represent feeders, sized by the number of outages.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with outage data including 'Feeder' column.
        feeder_coords (dict, optional): A dictionary mapping feeder names to [latitude, longitude].
                                        If None, simulated coordinates will be used.
        center_coords (list): [latitude, longitude] for the map's initial center (e.g., Rampur).
    """
    print("Generating interactive outage map...")
    
    if feeder_coords is None:
        # Simulate approximate coordinates for each feeder. In a real project, use actual GIS data.
        np.random.seed(43) # Ensure consistent simulated locations
        feeder_list = df['Feeder'].unique()
        feeder_coords = {
            f: [center_coords[0] + np.random.uniform(-0.05, 0.05), # Slightly vary lat/lon
                center_coords[1] + np.random.uniform(-0.05, 0.05)]
            for f in feeder_list
        }

    # Create a Folium map centered around Rampur
    m = folium.Map(location=center_coords, zoom_start=12)

    # Calculate outage counts per feeder to scale circle markers
    feeder_outage_counts = df['Feeder'].value_counts().reset_index()
    feeder_outage_counts.columns = ['Feeder', 'OutageCount']

    # Add circles for each feeder, scaled by outage count
    for index, row in feeder_outage_counts.iterrows():
        feeder = row['Feeder']
        count = row['OutageCount']
        coords = feeder_coords.get(feeder)
        if coords: # Ensure coordinates exist for the feeder
            folium.CircleMarker(
                location=coords,
                radius=np.sqrt(count) * 0.2, # Radius scaled by square root of count for better visual distinction
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=f'<b>{feeder}</b><br>Total Outages: {count}' # Popup info on click
            ).add_to(m)

    # Save the map as an HTML file in the 'output' directory
    map_path = 'output/rampur_outage_map.html'
    m.save(map_path)
    print(f"Interactive outage map saved to {map_path}")


def train_outage_prediction_model(df):
    """
    Trains a classification model to predict the daily occurrence of an outage.
    This function converts the event-based outage data into a daily time series
    suitable for classification (0 = no outage, 1 = at least one outage).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with outage and weather data.

    Returns:
        tuple: (trained_pipeline, daily_dataframe_with_predictions, test_set_indices)
    """
    print("Training power outage prediction model (Classification)...")

    # --- Prepare Daily Data for Classification ---
    # Create a DataFrame with all days in the dataset's range
    all_days = pd.date_range(start=df['StartTime'].min().floor('D'),
                             end=df['EndTime'].max().ceil('D'), freq='D')
    daily_df = pd.DataFrame({'ds': all_days}) # 'ds' for Prophet compatibility later
    daily_df['is_outage'] = 0 # Default: no outage on this day

    # Mark days with at least one outage as '1'
    outage_days = df['StartTime'].dt.floor('D').unique()
    daily_df.loc[daily_df['ds'].isin(outage_days), 'is_outage'] = 1

    # Aggregate weather data to daily averages/sums for merging
    weather_daily = df.groupby(df['StartTime'].dt.floor('D')).agg(
        Temperature_C=('Temperature_C', 'mean'),
        Humidity_=('Humidity_%', 'mean'),
        Rainfall_mm=('Rainfall_mm', 'sum'), # Sum rainfall over the day
        WindSpeed_mps=('WindSpeed_mps', 'max'), # Max wind speed for the day
        CloudCover_=('CloudCover_%', 'mean')
    ).reset_index().rename(columns={'StartTime': 'ds'})

    # Merge daily outage status with daily aggregated weather
    daily_df = pd.merge(daily_df, weather_daily, on='ds', how='left')

    # Fill any NaNs that might arise from days with no specific weather readings (unlikely if original weather is dense)
    for col in weather_daily.columns:
        if col != 'ds':
            daily_df[col] = daily_df[col].fillna(daily_df[col].mean()) # Impute with column mean

    # --- Feature Engineering for the Daily Prediction Model ---
    daily_df['day_of_week'] = daily_df['ds'].dt.dayofweek
    daily_df['month'] = daily_df['ds'].dt.month
    daily_df['year'] = daily_df['ds'].dt.year
    daily_df['day_of_year'] = daily_df['ds'].dt.dayofyear
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int) # 1 if Saturday/Sunday
    daily_df['is_monsoon'] = daily_df['month'].isin([6, 7, 8, 9]).astype(int) # Proxy for monsoon season in Rampur

    # Define features (X) and target (y) for the model
    features = [col for col in daily_df.columns if col not in ['ds', 'is_outage']]
    X = daily_df[features]
    y = daily_df['is_outage']

    # Handle categorical features for scikit-learn models using one-hot encoding
    X = pd.get_dummies(X, columns=['day_of_week', 'month'], drop_first=True) # drop_first to avoid multicollinearity

    # --- Split Data (Chronological Split) ---
    # It's crucial for time-series-like data to split chronologically to avoid data leakage.
    split_point = int(len(X) * 0.8) # 80% for training, 20% for testing
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # Store test indices to retrieve original dates later for plotting
    X_test_idx = X_test.index

    # --- Model Training Pipeline ---
    # Use a Pipeline for robust preprocessing and model application.
    # SimpleImputer: Handles any remaining missing values.
    # StandardScaler: Scales numerical features to have zero mean and unit variance.
    # RandomForestClassifier: The chosen classification model. `class_weight='balanced'` helps with imbalanced classes.
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class (outage)

    # --- Evaluate Model Performance ---
    print("\n--- Outage Occurrence Prediction Performance ---")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Outage', 'Outage'], yticklabels=['No Outage', 'Outage'])
    plt.title('Confusion Matrix for Outage Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC Curve Visualization (Receiver Operating Characteristic)
    # Shows the trade-off between True Positive Rate and False Positive Rate.
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Power outage prediction model training complete.")
    return model_pipeline, daily_df, X_test_idx # Return model, the daily df, and test indices


def forecast_outage_frequency(df):
    """
    Forecasts the overall frequency of outages using Facebook Prophet.
    Prophet is suitable for time series with strong seasonal components.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with outage data.

    Returns:
        tuple: (trained_prophet_model, forecast_dataframe)
    """
    print("Forecasting overall outage frequency using Prophet...")

    # Aggregate total outages per day. Prophet requires 'ds' (datetime) and 'y' (value).
    daily_outages = df.groupby(df['StartTime'].dt.floor('D'))['OutageID'].count().reset_index()
    daily_outages.columns = ['ds', 'y']

    # --- Prepare Regressors for Prophet (Daily Aggregated Weather) ---
    weather_daily_regressors = df.groupby(df['StartTime'].dt.floor('D')).agg(
        Temperature_C=('Temperature_C', 'mean'),
        Humidity_=('Humidity_%', 'mean'),
        Rainfall_mm=('Rainfall_mm', 'sum'),
        WindSpeed_mps=('WindSpeed_mps', 'max'),
        CloudCover_=('CloudCover_%', 'mean')
    ).reset_index().rename(columns={'StartTime': 'ds'})

    # Merge weather regressors into the daily outages DataFrame for Prophet
    prophet_df = pd.merge(daily_outages, weather_daily_regressors, on='ds', how='left')

    # Fill any NaNs in regressors (e.g., if some days had no outages and thus no weather entry)
    for col in ['Temperature_C', 'Humidity_', 'Rainfall_mm', 'WindSpeed_mps', 'CloudCover_']:
        prophet_df[col] = prophet_df[col].fillna(prophet_df[col].mean()) # Impute with mean

    # --- Initialize and Fit Prophet Model ---
    model = Prophet(
        yearly_seasonality=True,  # Account for annual patterns (e.g., summer heat, monsoon)
        weekly_seasonality=True,  # Account for weekly patterns (e.g., more outages on weekdays/weekends)
        daily_seasonality=False   # Our data is aggregated daily, so daily seasonality won't apply directly
    )

    # Add weather features as external regressors
    model.add_regressor('Temperature_C')
    model.add_regressor('Humidity_')
    model.add_regressor('Rainfall_mm')
    model.add_regressor('WindSpeed_mps')
    model.add_regressor('CloudCover_')

    model.fit(prophet_df)

    # --- Create Future DataFrame for Forecasting ---
    future = model.make_future_dataframe(periods=30) # Forecast 30 days into the future

    # --- Provide Regressor Values for the Future ---
    # For a real application, you would use actual weather forecasts for these future dates.
    # For this synthetic example, we'll extrapolate by repeating the last year's weather pattern.
    # This is a simplification and not ideal for real-world forecasting.
    last_year_data = weather_daily_regressors[weather_daily_regressors['ds'].dt.year == prophet_df['ds'].max().year - 1].set_index('ds')

    future_weather_data = []
    for i, dt in enumerate(future['ds']):
        target_date_last_year = dt - pd.DateOffset(years=1)
        
        # --- FIX for AttributeError: 'TimedeltaIndex' object has no attribute 'abs' ---
        # Calculate time differences and then apply np.abs() to the TimedeltaIndex
        time_diffs = last_year_data.index - target_date_last_year
        closest_weather = last_year_data.iloc[np.abs(time_diffs).argsort()[:1]]
        # --- END FIX ---
        
        if not closest_weather.empty:
            # drop 'ds' with errors='ignore' in case it's not present (e.g. if the index was already dropped)
            future_weather_data.append(closest_weather.iloc[0].drop('ds', errors='ignore').to_dict()) 
        else:
            # Fallback: if no historical match (e.g., very short history), use mean values
            future_weather_data.append({col: prophet_df[col].mean() for col in ['Temperature_C', 'Humidity_', 'Rainfall_mm', 'WindSpeed_mps', 'CloudCover_']})
    
    future_weather_df = pd.DataFrame(future_weather_data)
    # Assign the generated future regressor values to the 'future' DataFrame
    for col in ['Temperature_C', 'Humidity_', 'Rainfall_mm', 'WindSpeed_mps', 'CloudCover_']:
        future[col] = future_weather_df[col].values # Ensure length matches 'future' DataFrame

    # Generate the forecast
    forecast = model.predict(future)

    print("\n--- Outage Frequency Forecast ---")
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title('Outage Frequency Forecast (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Number of Outages')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot forecast components (trend, weekly, yearly seasonality, regressors)
    fig2 = model.plot_components(forecast)
    plt.suptitle('Outage Frequency Forecast Components', y=1.02) # Adjust suptitle position
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    print("Outage frequency forecasting complete.")
    return model, forecast