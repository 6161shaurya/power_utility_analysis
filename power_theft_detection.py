# power_theft_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import IsolationForest, RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.svm import OneClassSVM # Another anomaly detection method (less common for large datasets)
from imblearn.over_sampling import SMOTE # For handling imbalanced datasets (if using supervised learning)
from sklearn.linear_model import LogisticRegression # Example of a simple supervised classifier
from sklearn.neural_network import MLPClassifier # Example of a simple neural network classifier

# Set plotting style for better visualization aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7) # Default figure size for plots


def preprocess_consumption_data(consumption_df):
    """
    Cleans and engineers features for electricity consumption data.
    This prepares the data for theft detection models.

    Args:
        consumption_df (pd.DataFrame): DataFrame containing raw electricity consumption records.

    Returns:
        tuple: (pd.DataFrame, LabelEncoder_instance)
               Returns the preprocessed DataFrame and the fitted LabelEncoder for 'Connection_Type'.
    """
    print("Starting consumption data preprocessing...")
    
    # Ensure 'DateTime' column is in datetime format
    consumption_df['DateTime'] = pd.to_datetime(consumption_df['DateTime'])

    # --- Basic Time-based Feature Engineering ---
    consumption_df['Hour'] = consumption_df['DateTime'].dt.hour
    consumption_df['DayOfWeek'] = consumption_df['DateTime'].dt.dayofweek
    consumption_df['Month'] = consumption_df['DateTime'].dt.month
    consumption_df['DayOfYear'] = consumption_df['DateTime'].dt.dayofyear
    consumption_df['IsWeekend'] = consumption_df['DayOfWeek'].isin([5, 6]).astype(int) # 1 if Saturday/Sunday
    consumption_df['Quarter'] = consumption_df['DateTime'].dt.quarter

    # --- Lagged Features ---
    consumption_df = consumption_df.sort_values(by=['ConsumerID', 'DateTime'])
    
    consumption_df['Consumption_PrevHour'] = consumption_df.groupby('ConsumerID')['Consumption_kWh'].shift(1)
    consumption_df['Consumption_PrevDay'] = consumption_df.groupby('ConsumerID')['Consumption_kWh'].shift(24)

    # --- Rolling Window Features ---
    consumption_df['Consumption_24hr_Avg'] = consumption_df.groupby('ConsumerID')['Consumption_kWh'].transform(lambda x: x.rolling(window=24, min_periods=1).mean().shift(1))
    consumption_df['Consumption_24hr_Std'] = consumption_df.groupby('ConsumerID')['Consumption_kWh'].transform(lambda x: x.rolling(window=24, min_periods=1).std().shift(1))

    # --- Handle NaNs created by shifting/rolling ---
    consumption_df.fillna(0, inplace=True) 

    # --- Encode Categorical Features ---
    # LabelEncoder assigns a unique integer to each category.
    le = LabelEncoder() # <--- LabelEncoder is defined here
    consumption_df['Connection_Type_Encoded'] = le.fit_transform(consumption_df['Connection_Type'])

    print("Consumption data preprocessing complete.")
    return consumption_df, le # <--- NOW RETURNING 'le'


def perform_eda_consumption(df):
    """
    Performs Exploratory Data Analysis (EDA) on the preprocessed consumption data.
    Visualizes overall consumption distribution, patterns by connection type,
    and differences between theft and non-theft consumption if labels are available.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with consumption data.
    """
    print("Performing EDA for power consumption and theft patterns...")

    # 1. Overall Consumption Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Consumption_kWh'], bins=50, kde=True)
    plt.title('Distribution of Hourly Electricity Consumption (kWh)')
    plt.xlabel('Consumption (kWh)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Consumption by Connection Type (using Box Plot to show distribution)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Connection_Type', y='Consumption_kWh', palette='pastel', showfliers=False)
    plt.title('Hourly Consumption by Connection Type')
    plt.xlabel('Connection Type')
    plt.ylabel('Consumption (kWh)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 3. Average Daily/Hourly Consumption Patterns by Connection Type
    df['Date'] = df['DateTime'].dt.date
    daily_avg_consumption = df.groupby(['Date', 'Connection_Type'])['Consumption_kWh'].sum().reset_index()
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=daily_avg_consumption, x='Date', y='Consumption_kWh', hue='Connection_Type')
    plt.title('Daily Total Consumption by Connection Type Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Consumption (kWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    hourly_avg_consumption = df.groupby(['Hour', 'Connection_Type'])['Consumption_kWh'].mean().reset_index()
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=hourly_avg_consumption, x='Hour', y='Consumption_kWh', hue='Connection_Type')
    plt.title('Average Hourly Consumption Pattern by Connection Type')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Consumption (kWh)')
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Theft vs. Non-Theft Consumption Comparison (if 'IsTheft' column exists)
    if 'IsTheft' in df.columns and df['IsTheft'].any():
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x='IsTheft', y='Consumption_kWh', palette='coolwarm', showfliers=False)
        plt.title('Consumption Distribution: Non-Theft vs. Theft (Labeled Data)')
        plt.xlabel('Is Theft (0: No, 1: Yes)')
        plt.ylabel('Consumption (kWh)')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        theft_consumers_sample = df[df['IsTheft'] == True]['ConsumerID'].unique()[:3]
        non_theft_consumers_sample = df[df['IsTheft'] == False]['ConsumerID'].unique()[:3]

        num_sample_plots = len(theft_consumers_sample) + len(non_theft_consumers_sample)
        if num_sample_plots > 0:
            fig, axes = plt.subplots(num_sample_plots, 1, figsize=(15, 4 * num_sample_plots))
            if not isinstance(axes, np.ndarray):
                axes = [axes]

            for i, cid in enumerate(theft_consumers_sample):
                consumer_data = df[df['ConsumerID'] == cid].sample(n=min(500, len(df[df['ConsumerID'] == cid])), random_state=42).sort_values('DateTime')
                sns.lineplot(data=consumer_data, x='DateTime', y='Consumption_kWh', ax=axes[i], color='red', label='Theft')
                axes[i].set_title(f'Consumer {cid} (Theft) Consumption Profile')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Consumption (kWh)')
                axes[i].legend()
                axes[i].grid(True)

            for j, cid in enumerate(non_theft_consumers_sample):
                consumer_data = df[df['ConsumerID'] == cid].sample(n=min(500, len(df[df['ConsumerID'] == cid])), random_state=42).sort_values('DateTime')
                sns.lineplot(data=consumer_data, x='DateTime', y='Consumption_kWh', ax=axes[len(theft_consumers_sample) + j], color='green', label='Non-Theft')
                axes[len(theft_consumers_sample) + j].set_ylabel('Consumption (kWh)')
                axes[len(theft_consumers_sample) + j].legend()
                axes[len(theft_consumers_sample) + j].grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("Not enough sample consumers to plot consumption profiles.")

    else:
        print("No 'IsTheft' labels found or no theft cases to perform comparison EDA.")

    print("EDA for consumption data complete.")


def train_theft_detection_model_supervised(df, le_encoder=None): # <--- ADDED le_encoder as an argument
    """
    Trains a supervised classification model for power theft detection.
    This model learns from historical data with known theft labels ('IsTheft').

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with consumption data, including 'IsTheft' labels.
        le_encoder (LabelEncoder, optional): The LabelEncoder instance used for 'Connection_Type'.
                                             This is returned to ensure consistency for future predictions.

    Returns:
        tuple: (trained_model, StandardScaler_instance, LabelEncoder_instance)
               Returns None, None, None if 'IsTheft' column is missing or empty.
    """
    print("Training supervised power theft detection model...")

    if 'IsTheft' not in df.columns or not df['IsTheft'].any():
        print("Warning: 'IsTheft' column not found or no theft cases. Cannot train supervised model.")
        print("Please ensure your data includes labeled theft cases for supervised learning.")
        return None, None, None

    features = [
        'SanctionedLoad_kW', 'Consumption_kWh', 'Consumption_PrevHour',
        'Consumption_PrevDay', 'Consumption_24hr_Avg', 'Consumption_24hr_Std',
        'Hour', 'DayOfWeek', 'Month', 'DayOfYear', 'IsWeekend', 'Quarter',
        'Connection_Type_Encoded'
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df['IsTheft']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Original training set distribution (0:No Theft, 1:Theft): {y_train.value_counts(normalize=True)}")
    print(f"Original test set distribution (0:No Theft, 1:Theft): {y_test.value_counts(normalize=True)}")

    if y_train.sum() > 0 and y_train.sum() < len(y_train) * 0.5:
        print("Applying SMOTE to balance the training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Resampled training set distribution: {y_train_resampled.value_counts(normalize=True)}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("SMOTE not applied (data already balanced or no minority class).")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Supervised Theft Detection Performance ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Theft', 'Theft'], yticklabels=['Not Theft', 'Theft'])
    plt.title('Confusion Matrix for Supervised Theft Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Theft Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Theft Detection')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Supervised power theft detection model training complete.")
    return model, scaler, le_encoder # <--- NOW RETURNING le_encoder received as argument


def detect_theft_unsupervised(df, contamination=0.05):
    """
    Detects power theft using an unsupervised anomaly detection model (Isolation Forest).
    This method does not require pre-labeled theft cases for training,
    but we can use 'IsTheft' (if available) for evaluation.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with consumption data.
        contamination (float): The proportion of outliers in the data set.
                               Used to set the threshold for anomaly detection.

    Returns:
        tuple: (DataFrame_with_anomaly_predictions, trained_IsolationForest_model, StandardScaler_instance)
    """
    print(f"Detecting power theft using unsupervised anomaly detection (Isolation Forest) with contamination={contamination}...")

    features = [
        'SanctionedLoad_kW', 'Consumption_kWh', 'Consumption_PrevHour',
        'Consumption_PrevDay', 'Consumption_24hr_Avg', 'Consumption_24hr_Std',
        'Hour', 'DayOfWeek', 'Month', 'DayOfYear', 'IsWeekend', 'Quarter',
        'Connection_Type_Encoded'
    ]
    features = [f for f in features if f in df.columns]

    X_anomaly = df[features].copy()
    X_anomaly.fillna(0, inplace=True) 

    scaler_anomaly = StandardScaler()
    X_anomaly_scaled = scaler_anomaly.fit_transform(X_anomaly)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X_anomaly_scaled)

    df['Anomaly_Score'] = iso_forest.decision_function(X_anomaly_scaled)
    df['IsAnomaly_Predicted'] = iso_forest.predict(X_anomaly_scaled)

    df['IsAnomaly_Predicted'] = df['IsAnomaly_Predicted'].map({1: 0, -1: 1})

    print("\n--- Unsupervised Theft Detection Results (Isolation Forest) ---")
    detected_anomalies_count = df['IsAnomaly_Predicted'].sum()
    print(f"Total anomalies detected: {detected_anomalies_count} out of {len(df)} records.")

    if 'IsTheft' in df.columns and df['IsTheft'].any():
        print("\nEvaluation against true 'IsTheft' labels:")
        print(classification_report(df['IsTheft'], df['IsAnomaly_Predicted']))

        cm = confusion_matrix(df['IsTheft'], df['IsAnomaly_Predicted'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Theft', 'Theft'], yticklabels=['Not Theft', 'Theft'])
        plt.title('Confusion Matrix for Unsupervised Theft Detection (vs. True Labels)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.histplot(df[df['IsTheft'] == 0]['Anomaly_Score'], color='green', label='Not Theft', kde=True, stat='density', alpha=0.5)
        sns.histplot(df[df['IsTheft'] == 1]['Anomaly_Score'], color='red', label='Theft', kde=True, stat='density', alpha=0.5)
        plt.title('Anomaly Scores Distribution: True Non-Theft vs. True Theft')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No 'IsTheft' labels found in the dataset for evaluation against true cases.")

    print("Unsupervised power theft detection complete.")
    return df, iso_forest, scaler_anomaly