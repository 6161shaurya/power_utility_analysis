# dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster # For clustering markers on the map
import plotly.express as px # For interactive plots

# Import functions from your modules
# Assuming these files are in the same directory as dashboard_app.py
from power_outage_prediction import preprocess_outage_data, train_outage_prediction_model, forecast_outage_frequency, visualize_outage_map, perform_eda_outages
from power_theft_detection import preprocess_consumption_data, train_theft_detection_model_supervised, detect_theft_unsupervised, perform_eda_consumption

# Required for displaying folium maps in Streamlit
try:
    from streamlit_folium import st_folium
except ImportError:
    st.error("Please install `streamlit-folium` to enable map visualization: `pip install streamlit-folium`")
    # Define a dummy function to avoid errors if not installed
    def st_folium(*args, **kwargs):
        st.write("`streamlit-folium` not installed. Cannot display map.")
        return None

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Rampur Power Utility Analytics")

# --- Global Variables / Data Loading (Cached for Performance) ---
# Use st.cache_data to cache data loading. This prevents reloading data every time
# the script reruns (e.g., when a widget is interacted with).
@st.cache_data
def load_data():
    """
    Loads raw data from CSV files and performs initial preprocessing.
    """
    try:
        outages_df = pd.read_csv('data/raw_outages.csv')
        weather_df = pd.read_csv('data/raw_weather.csv')
        consumption_df = pd.read_csv('data/raw_consumption.csv')
        
        # Ensure DateTime columns are parsed as datetime objects right after loading
        outages_df['StartTime'] = pd.to_datetime(outages_df['StartTime'])
        outages_df['EndTime'] = pd.to_datetime(outages_df['EndTime'])
        weather_df['DateTime'] = pd.to_datetime(weather_df['DateTime'])
        consumption_df['DateTime'] = pd.to_datetime(consumption_df['DateTime'])

    except FileNotFoundError:
        st.error("Data files not found in the 'data/' directory. Please run `python main.py` first to generate the synthetic data.")
        st.stop() # Stop the Streamlit app execution
    
    return outages_df, weather_df, consumption_df

# Load raw data globally
outages_df_raw, weather_df_raw, consumption_df_raw = load_data()


# --- Data Preprocessing (Cached for Performance) ---
# Preprocess data once and store in Streamlit's session state.
# This prevents re-running potentially expensive preprocessing on every user interaction.
# Using deep copies (.copy()) to ensure original raw dataframes are not modified.
if 'processed_outages' not in st.session_state:
    st.session_state.processed_outages = preprocess_outage_data(outages_df_raw.copy(), weather_df_raw.copy())
    
# --- FIX STARTS HERE ---
# Unpack the tuple returned by preprocess_consumption_data
if 'processed_consumption' not in st.session_state:
    processed_df, label_encoder_instance = preprocess_consumption_data(consumption_df_raw.copy())
    st.session_state.processed_consumption = processed_df
    st.session_state.consumption_label_encoder = label_encoder_instance # Store label encoder too, if needed later
# --- FIX ENDS HERE ---

# Access preprocessed data from session state
processed_outages = st.session_state.processed_outages
processed_consumption = st.session_state.processed_consumption


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# Radio buttons for navigating between different sections/pages
page = st.sidebar.radio("Go to", ["Home", "Power Outage Analysis", "Power Theft Analysis", "About"])


# --- Home Page Content ---
if page == "Home":
    st.title("Rampur Power Utility Analytics Dashboard")
    st.markdown("""
    Welcome to the **Rampur Power Utility Analytics Dashboard**. This interactive tool
    leverages data science to help analyze and predict power outages, and detect
    potential electricity theft.
    """)

    st.subheader("Current Data Overview")
    # Display key metrics using Streamlit's st.metric
    col1, col2, col3 = st.columns(3) # Create three columns for metrics
    with col1:
        st.metric("Total Outages Records", f"{len(outages_df_raw):,}")
        st.metric("Total Unique Feeders", f"{outages_df_raw['Feeder'].nunique():,}")
    with col2:
        st.metric("Total Consumption Records", f"{len(consumption_df_raw):,}")
        st.metric("Total Unique Consumers", f"{consumption_df_raw['ConsumerID'].nunique():,}")
    with col3:
        # Check if 'IsTheft' column exists in consumption data (it should for synthetic data)
        if 'IsTheft' in consumption_df_raw.columns:
            # Count records marked as theft
            st.metric("Known Theft Records", f"{consumption_df_raw['IsTheft'].sum():,}")
            # Count unique consumers with at least one theft record
            theft_consumers_count = consumption_df_raw[consumption_df_raw['IsTheft'] == True]['ConsumerID'].nunique()
            st.metric("Unique Known Theft Consumers", f"{theft_consumers_count:,}")
        else:
            st.metric("Known Theft Cases (Records)", "N/A (No labels)")

    st.write("---")
    st.subheader("Project Goals & Potential Impact in Rampur")
    st.markdown("""
    This project aims to leverage data science to:
    - **Improve Grid Reliability:** By predicting outages, maintenance teams can be deployed proactively, reducing downtime for Rampur residents.
    - **Reduce Financial Losses:** By identifying theft, the electricity utility can recover lost revenue, leading to better infrastructure investment.
    - **Enhance Customer Satisfaction:** Fewer outages and fairer billing practices contribute to a more positive experience for consumers in Rampur.
    - **Optimize Resource Allocation:** Better insights lead to more efficient use of personnel, equipment, and budget for the local power distribution company.
    """)
    st.info("Note: All data used in this dashboard is synthetic and generated for demonstration purposes. Real-world application requires actual data.")


# --- Power Outage Analysis Page Content ---
elif page == "Power Outage Analysis":
    st.title("Power Outage Analysis & Prediction")
    st.write("Explore historical outage patterns, understand their causes, and generate predictive insights for better grid management.")

    # Tabs for organizing content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA & Overview", "🗺️ Geospatial Analysis", "🧠 Prediction Model", "📈 Outage Frequency Forecast"])

    with tab1:
        st.subheader("Exploratory Data Analysis (EDA) of Power Outages")
        st.write("Dive into the characteristics of past power outages in Rampur.")

        # Outage Frequency Over Time (Monthly)
        st.markdown("#### Monthly Power Outage Frequency")
        monthly_outages = processed_outages.set_index('StartTime').resample('ME')['OutageID'].count()
        fig_monthly = px.line(monthly_outages, x=monthly_outages.index, y='OutageID',
                              labels={'OutageID': 'Number of Outages', 'index': 'Date'},
                              title='Monthly Power Outage Frequency')
        st.plotly_chart(fig_monthly, use_container_width=True)

        col1_eda, col2_eda = st.columns(2) # Two columns for side-by-side plots
        with col1_eda:
            # Outage Duration Distribution
            st.markdown("#### Distribution of Outage Durations")
            fig_duration = px.histogram(processed_outages, x='DurationMinutes', nbins=50,
                                        title='Distribution of Outage Durations (Minutes)',
                                        labels={'DurationMinutes': 'Duration (Minutes)', 'count': 'Number of Outages'})
            st.plotly_chart(fig_duration, use_container_width=True)
        with col2_eda:
            # Outage Causes Distribution
            st.markdown("#### Distribution of Outage Causes")
            cause_counts = processed_outages['Cause'].value_counts().reset_index()
            cause_counts.columns = ['Cause', 'Count']
            fig_causes = px.bar(cause_counts, y='Cause', x='Count', orientation='h',
                                title='Distribution of Outage Causes')
            st.plotly_chart(fig_causes, use_container_width=True)

        st.markdown("#### Outage Frequency by Time of Day and Day of Week")
        fig_hourly = px.histogram(processed_outages, x='OutageHour', nbins=24, title='Outage Frequency by Hour of Day',
                                  labels={'OutageHour': 'Hour of Day', 'count': 'Number of Outages'})
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Mapping day of week numbers to names for better readability
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        processed_outages['OutageDayOfWeek_Name'] = processed_outages['OutageDayOfWeek'].map(day_names)
        fig_daily = px.histogram(processed_outages, x='OutageDayOfWeek_Name', category_orders={'OutageDayOfWeek_Name': list(day_names.values())},
                                 title='Outage Frequency by Day of Week',
                                 labels={'OutageDayOfWeek_Name': 'Day of Week', 'count': 'Number of Outages'})
        st.plotly_chart(fig_daily, use_container_width=True)


    with tab2:
        st.subheader("Geospatial Analysis of Outages")
        st.write("Visualize outage hotspots on a map. (Feeder locations are simulated for demonstration).")

        # Simulate approximate coordinates for each feeder (replace with actual GIS data if available)
        center_coords = [28.837, 79.037] # Approximate center of Rampur city
        # Ensure consistent simulation of feeder coords by using a fixed random seed
        np.random.seed(43)
        feeder_list = processed_outages['Feeder'].unique()
        feeder_coords = {
            f: [center_coords[0] + np.random.uniform(-0.05, 0.05),
                center_coords[1] + np.random.uniform(-0.05, 0.05)]
            for f in feeder_list
        }

        # Create Folium map
        m = folium.Map(location=center_coords, zoom_start=12)

        feeder_outage_counts = processed_outages['Feeder'].value_counts().reset_index()
        feeder_outage_counts.columns = ['Feeder', 'OutageCount']

        # Add circles for each feeder, scaled by outage count
        for index, row in feeder_outage_counts.iterrows():
            feeder = row['Feeder']
            count = row['OutageCount']
            coords = feeder_coords.get(feeder)
            if coords:
                folium.CircleMarker(
                    location=coords,
                    radius=np.sqrt(count) * 0.2, # Scale radius by sqrt of count for better visualization
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6,
                    popup=f'<b>{feeder}</b><br>Total Outages: {count}'
                ).add_to(m)
        
        st.markdown("#### Outage Hotspots Map (Simulated Feeder Locations)")
        st_info_text = "This map displays simulated feeder locations with circles scaled by the number of outages. Larger red circles indicate more frequent outages in that simulated area."
        st.info(st_info_text)
        st_folium(m, width=900, height=500)


    with tab3:
        st.subheader("Outage Prediction Model (Classification)")
        st.write("This model predicts whether an outage will occur on a given day based on historical data and weather conditions. It's a classification task (Outage / No Outage).")

        # Button to trigger model training
        if st.button("Train Outage Prediction Model"):
            with st.spinner("Training model... This might take a moment based on data size."):
                # Call the training function from power_outage_prediction.py
                model, daily_df_with_preds, X_test_idx = train_outage_prediction_model(processed_outages.copy())
                # Store the trained model and related data in session state for later use
                st.session_state.outage_pred_model = model
                st.session_state.daily_df_with_preds = daily_df_with_preds
                st.session_state.outage_X_test_idx = X_test_idx
            st.success("Outage Prediction Model Trained Successfully!")
            st.write("Check your terminal/console for detailed performance metrics (Classification Report, Confusion Matrix, ROC Curve).")

        # Display results if model has been trained
        if 'outage_pred_model' in st.session_state:
            st.markdown("#### Model Insights")
            model = st.session_state.outage_pred_model
            daily_df = st.session_state.daily_df_with_preds
            X_test_idx = st.session_state.outage_X_test_idx

            # Feature Importance
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                st.subheader("Feature Importance for Outage Prediction")
                feature_importances = model.named_steps['classifier'].feature_importances_
                
                temp_daily_df_for_features = daily_df.copy()
                temp_X_for_features = temp_daily_df_for_features[[col for col in temp_daily_df_for_features.columns if col not in ['ds', 'is_outage']]]
                temp_X_for_features = pd.get_dummies(temp_X_for_features, columns=['day_of_week', 'month'], drop_first=True)
                feature_names = temp_X_for_features.columns
                
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
                feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
                fig_feat_imp = px.bar(feat_imp_df.head(15), x='Importance', y='Feature', orientation='h',
                                      title='Top 15 Features Influencing Outage Prediction')
                st.plotly_chart(fig_feat_imp, use_container_width=True)
            
            # Display actual vs predicted for a portion of test set
            st.subheader("Actual vs. Predicted Outages (Test Set Sample)")
            test_predictions_df = daily_df.loc[X_test_idx].copy().set_index('ds')
            
            temp_X_test_for_pred = daily_df.loc[X_test_idx].copy()
            temp_X_test_for_pred = temp_X_test_for_pred[[col for col in temp_X_test_for_pred.columns if col not in ['ds', 'is_outage']]]
            temp_X_test_for_pred = pd.get_dummies(temp_X_test_for_pred, columns=['day_of_week', 'month'], drop_first=True)
            
            train_cols_from_scaler = model.named_steps['scaler'].feature_names_in_
            missing_cols = set(train_cols_from_scaler) - set(temp_X_test_for_pred.columns)
            for c in missing_cols:
                temp_X_test_for_pred[c] = 0
            temp_X_test_for_pred = temp_X_test_for_pred[train_cols_from_scaler] # Ensure order

            temp_X_test_scaled = model.named_steps['scaler'].transform(temp_X_test_for_pred)
            test_predictions_df['Predicted_is_outage'] = model.predict(temp_X_test_scaled)
            
            plot_df = test_predictions_df.reset_index()
            fig_pred_actual = px.line(plot_df, x='ds', y='is_outage', title='Actual Outages (Test Set)',
                                      labels={'ds': 'Date', 'is_outage': 'Outage Occurred (1=Yes)'})
            fig_pred_actual.add_scatter(x=plot_df['ds'], y=plot_df['Predicted_is_outage'], mode='lines', name='Predicted Outages',
                                        line=dict(dash='dot', color='red'))
            fig_pred_actual.update_layout(hovermode="x unified") # Enhanced hover info
            st.plotly_chart(fig_pred_actual, use_container_width=True)


    with tab4:
        st.subheader("Outage Frequency Forecasting (Prophet)")
        st.write("This section forecasts the general trend of outages for the next 30 days using the Prophet library, taking into account yearly and weekly seasonality, and weather regressors.")

        # Button to trigger forecast generation
        if st.button("Generate Outage Frequency Forecast"):
            with st.spinner("Generating forecast... This might take a moment."):
                model_prophet, forecast_df = forecast_outage_frequency(processed_outages.copy())
                st.session_state.prophet_model = model_prophet
                st.session_state.prophet_forecast = forecast_df
            st.success("Outage Frequency Forecast Generated!")

        # Display forecast if generated
        if 'prophet_forecast' in st.session_state:
            st.markdown("#### Forecast Plot")
            fig_forecast = st.session_state.prophet_model.plot(st.session_state.prophet_forecast)
            st.pyplot(fig_forecast) # Display matplotlib figure in Streamlit

            st.markdown("#### Forecast Components")
            fig_components = st.session_state.prophet_model.plot_components(st.session_state.prophet_forecast)
            st.pyplot(fig_components)


# --- Power Theft Analysis Page Content ---
elif page == "Power Theft Analysis":
    st.title("Power Theft Analysis & Detection")
    st.write("Identify anomalous consumption patterns that may indicate power theft.")

    # Tabs for organizing content
    tab1, tab2, tab3 = st.tabs(["📊 EDA & Consumption Patterns", "🧠 Supervised Theft Detection", "🕵️ Unsupervised Theft Detection"])

    with tab1:
        st.subheader("Exploratory Data Analysis (EDA) of Consumption Data")
        st.write("Understand typical electricity consumption patterns and how they differ across customer types.")

        st.markdown("#### Average Hourly Consumption by Connection Type")
        # Ensure processed_consumption is correctly accessed from session_state
        # It is already extracted as processed_consumption = st.session_state.processed_consumption at the top
        hourly_avg_consumption = processed_consumption.groupby(['Hour', 'Connection_Type'])['Consumption_kWh'].mean().reset_index()
        fig_hourly_cons = px.line(hourly_avg_consumption, x='Hour', y='Consumption_kWh', color='Connection_Type',
                                  title='Average Hourly Consumption Pattern by Connection Type',
                                  labels={'Consumption_kWh': 'Avg. Consumption (kWh)'})
        st.plotly_chart(fig_hourly_cons, use_container_width=True)

        # Only show theft vs non-theft plots if 'IsTheft' labels are available
        if 'IsTheft' in processed_consumption.columns and processed_consumption['IsTheft'].any():
            st.markdown("#### Consumption Distribution: Non-Theft vs. Theft")
            fig_theft_dist = px.box(processed_consumption, x='IsTheft', y='Consumption_kWh',
                                    title='Consumption Distribution: Non-Theft vs. Theft (0: No, 1: Yes)',
                                    labels={'IsTheft': 'Is Theft', 'Consumption_kWh': 'Consumption (kWh)'})
            st.plotly_chart(fig_theft_dist, use_container_width=True)

            st.markdown("#### Sample Consumer Consumption Profiles")
            theft_consumers_sample = processed_consumption[processed_consumption['IsTheft'] == True]['ConsumerID'].unique()[:2]
            non_theft_consumers_sample = processed_consumption[processed_consumption['IsTheft'] == False]['ConsumerID'].unique()[:2]

            num_sample_plots = len(theft_consumers_sample) + len(non_theft_consumers_sample)
            if num_sample_plots > 0:
                fig, axes = plt.subplots(num_sample_plots, 1, figsize=(15, 4 * num_sample_plots))
                if not isinstance(axes, np.ndarray):
                    axes = [axes]

                for i, cid in enumerate(theft_consumers_sample):
                    consumer_data = processed_consumption[processed_consumption['ConsumerID'] == cid].sample(n=min(500, len(processed_consumption[processed_consumption['ConsumerID'] == cid])), random_state=42).sort_values('DateTime')
                    sns.lineplot(data=consumer_data, x='DateTime', y='Consumption_kWh', ax=axes[i], color='red', label='Theft')
                    axes[i].set_title(f'Consumer {cid} (Theft) Consumption Profile')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Consumption (kWh)')
                    axes[i].legend()
                    axes[i].grid(True)

                for j, cid in enumerate(non_theft_consumers_sample):
                    consumer_data = processed_consumption[processed_consumption['ConsumerID'] == cid].sample(n=min(500, len(processed_consumption[processed_consumption['ConsumerID'] == cid])), random_state=42).sort_values('DateTime')
                    sns.lineplot(data=consumer_data, x='DateTime', y='Consumption_kWh', ax=axes[len(theft_consumers_sample) + j], color='green', label='Non-Theft')
                    axes[len(theft_consumers_sample) + j].set_ylabel('Consumption (kWh)')
                    axes[len(theft_consumers_sample) + j].legend()
                    axes[len(theft_consumers_sample) + j].grid(True)

                plt.tight_layout()
                plt.show()
            else:
                st.info("Not enough sample consumers to plot consumption profiles.")

        else:
            st.info("No 'IsTheft' labels found in data for comparison plots.")


    with tab2:
        st.subheader("Supervised Theft Detection")
        st.write("This model learns from historical data with **labeled theft cases** to predict new ones. It is trained to classify consumption records as 'theft' or 'non-theft'.")
        st.info("NOTE: Our synthetic data includes these 'IsTheft' labels for demonstration. In real-world scenarios, obtaining such labels is a major challenge.")

        if st.button("Train Supervised Theft Detection Model"):
            if 'IsTheft' in processed_consumption.columns and processed_consumption['IsTheft'].any():
                with st.spinner("Training supervised model... This may take a few moments."):
                    # Pass the stored LabelEncoder instance
                    model, scaler, le_trained = train_theft_detection_model_supervised(processed_consumption.copy(), le_encoder=st.session_state.consumption_label_encoder)
                    st.session_state.supervised_theft_model = model
                    st.session_state.supervised_theft_scaler = scaler
                    st.session_state.supervised_theft_le = le_trained # Store it
                st.success("Supervised Theft Detection Model Trained Successfully!")
                st.write("Check your terminal/console for detailed performance metrics (Classification Report, Confusion Matrix, ROC/PR Curves).")
            else:
                st.warning("Cannot train supervised model: No 'IsTheft' labels found or no theft cases in the data. Supervised learning requires labeled data.")

        if 'supervised_theft_model' in st.session_state:
            st.markdown("#### Model Insights")
            model = st.session_state.supervised_theft_model
            
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance for Supervised Theft Detection")
                feature_importances = model.feature_importances_
                
                features_used = [
                    'SanctionedLoad_kW', 'Consumption_kWh', 'Consumption_PrevHour',
                    'Consumption_PrevDay', 'Consumption_24hr_Avg', 'Consumption_24hr_Std',
                    'Hour', 'DayOfWeek', 'Month', 'DayOfYear', 'IsWeekend', 'Quarter',
                    'Connection_Type_Encoded'
                ]
                
                feat_imp_df = pd.DataFrame({'Feature': features_used, 'Importance': feature_importances})
                feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
                fig_feat_imp_theft = px.bar(feat_imp_df.head(15), x='Importance', y='Feature', orientation='h',
                                            title='Top 15 Features Influencing Supervised Theft Detection')
                st.plotly_chart(fig_feat_imp_theft, use_container_width=True)


    with tab3:
        st.subheader("Unsupervised Theft Detection (Anomaly Detection)")
        st.write("This model identifies unusual consumption patterns without needing prior labels of theft. It's useful for discovering new or unknown theft methods.")
        st.info("The model will identify a percentage of records as 'anomalies' based on your selected 'Contamination Level'.")

        contamination_level = st.slider("Select Contamination Level (Estimated % of Anomalies)", 0.01, 0.10, 0.05, 0.01)

        if st.button("Run Unsupervised Theft Detection"):
            with st.spinner(f"Detecting anomalies with contamination={contamination_level}..."):
                processed_consumption_with_anomalies, iso_forest_model, iso_scaler = detect_theft_unsupervised(
                    processed_consumption.copy(), contamination=contamination_level
                )
                st.session_state.unsupervised_theft_results = processed_consumption_with_anomalies
                st.session_state.iso_forest_model = iso_forest_model
                st.session_state.iso_scaler = iso_scaler
            st.success("Anomaly Detection Complete!")

        if 'unsupervised_theft_results' in st.session_state:
            st.markdown("#### Detected Anomalies Overview")
            anomalies_df = st.session_state.unsupervised_theft_results[
                st.session_state.unsupervised_theft_results['IsAnomaly_Predicted'] == 1
            ]
            
            st.write(f"Found **{len(anomalies_df):,}** anomalous consumption records out of {len(processed_consumption_with_anomalies):,} total.")
            
            if not anomalies_df.empty:
                st.write("Top 10 most anomalous records (lowest anomaly score):")
                display_cols = ['ConsumerID', 'DateTime', 'Consumption_kWh', 'Anomaly_Score']
                if 'IsTheft' in anomalies_df.columns:
                    display_cols.append('IsTheft')
                
                st.dataframe(anomalies_df[display_cols].sort_values('Anomaly_Score').head(10))
                st.info("Records with lower anomaly scores are considered more anomalous/outlier-like.")
                
                st.markdown("#### Anomaly Scores Distribution")
                fig_anomaly_scores = px.histogram(st.session_state.unsupervised_theft_results, x='Anomaly_Score', color='IsAnomaly_Predicted',
                                                  title='Distribution of Anomaly Scores (Predicted Anomalies in Red)',
                                                  labels={'Anomaly_Score': 'Anomaly Score', 'IsAnomaly_Predicted': 'Predicted Anomaly (0=No, 1=Yes)'})
                st.plotly_chart(fig_anomaly_scores, use_container_width=True)

                if 'IsTheft' in anomalies_df.columns:
                    actual_theft_in_anomalies = anomalies_df['IsTheft'].sum()
                    st.write(f"Of the detected {len(anomalies_df):,} anomalies, **{actual_theft_in_anomalies:,}** records were actual theft cases (based on synthetic labels).")
                    st.info("This comparison gives an idea of how well the unsupervised model found the 'true' theft cases in our synthetic data.")
            else:
                st.info("No anomalies detected with the current contamination level. Try adjusting the slider or check your data.")


# --- About Page Content ---
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    This project is a demonstration of how data science and machine learning can be applied
    to address real-world challenges faced by power utility companies, using **Rampur** as a
    case study with **synthetic data**.

    **Key Components & Technologies:**
    - **Data Generation:** Python with NumPy and Pandas for creating simulated power outage,
      weather, and electricity consumption datasets.
    - **Data Preprocessing & Feature Engineering:** Pandas for data manipulation, cleaning,
      and creating new informative features from raw data.
    - **Exploratory Data Analysis (EDA):** Matplotlib, Seaborn, and Plotly Express for
      generating static and interactive visualizations to understand data patterns.
    - **Power Outage Prediction:**
        - **Random Forest Classifier:** A supervised machine learning model for predicting
          the *occurrence* of an outage (classification).
        - **Facebook Prophet:** A robust time series forecasting model for predicting the
          overall *frequency* of outages.
    - **Power Theft Detection:**
        - **Random Forest Classifier (Supervised):** Trained on labeled data to classify
          consumption as 'theft' or 'non-theft'.
        - **Isolation Forest (Unsupervised):** An anomaly detection algorithm used to
          identify unusual consumption patterns without requiring pre-labeled theft data.
    - **Interactive Dashboard:** **Streamlit** for building this user-friendly web interface,
      allowing users to explore data, trigger models, and view results interactively.
    - **Geospatial Visualization:** **Folium** and `streamlit-folium` for plotting data on interactive maps.

    **Disclaimer:**
    The data used in this dashboard is entirely **synthetic** and does not represent
    actual power consumption or outage data from Rampur or UPPCL. The models are
    trained solely on this simulated data and serve as a proof-of-concept.
    Real-world implementation would require actual, granular data, extensive
    domain expertise, and careful validation.

    **Developed by:** Shaurya Jain
    """)