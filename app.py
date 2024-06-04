import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# st.markdown(f"""
# <style>
# {open('style.css', 'r').read()}
# </style>
# """, unsafe_allow_html=True)
# Load data
data = pd.read_csv('machinery_data.csv')
data.fillna(method='ffill', inplace=True)

# Feature selection and normalization
features = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']
target_rul = 'RUL'
target_maintenance = 'maintenance'
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split data for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data[features], data[target_rul], test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data[features], data[target_maintenance], test_size=0.2, random_state=42)

# Train models
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

# Prediction function
def predict_maintenance(features):
    rul_pred = reg_model.predict([features])
    maint_pred = clf_model.predict([features])
    cluster_pred = kmeans.predict([features])
    return {
        'RUL Prediction': rul_pred[0],
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'Normal',
        'Anomaly Detection': 'Anomaly' if cluster_pred[0] == 1 else 'Normal'
    }

# Streamlit Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Historical Data", "Input Data", "Results", "Visualizations"],
        icons=["house", "table", "input-cursor", "check2-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.title("Welcome to the Predictive Maintenance Dashboard")
    st.markdown("""
    This application provides predictive maintenance insights for industrial machinery. 
    Use the navigation menu to explore different sections of the app.
    """)

elif selected == "Historical Data":
    st.title("📂 Historical Data")
    st.write(data.head(10))

elif selected == "Input Data":
    st.title("🔧 Input Features")
    st.markdown("Use the sliders to input the sensor readings and operational hours or generate random values.")

    if 'generated_values' not in st.session_state:
        st.session_state['generated_values'] = None

    if st.button('Generate Random Values'):
        sensor_1 = np.random.uniform(data['sensor_1'].min(), data['sensor_1'].max())
        sensor_2 = np.random.uniform(data['sensor_2'].min(), data['sensor_2'].max())
        sensor_3 = np.random.uniform(data['sensor_3'].min(), data['sensor_3'].max())
        operational_hours = np.random.uniform(data['operational_hours'].min(), data['operational_hours'].max())
        st.session_state['generated_values'] = [sensor_1, sensor_2, sensor_3, operational_hours]
        st.success("Random values generated successfully!")

    if st.session_state['generated_values'] is not None:
        st.write("**Generated Values:**")
        st.write(f"Sensor 1: {st.session_state['generated_values'][0]:.2f}")
        st.write(f"Sensor 2: {st.session_state['generated_values'][1]:.2f}")
        st.write(f"Sensor 3: {st.session_state['generated_values'][2]:.2f}")
        st.write(f"Operational Hours: {st.session_state['generated_values'][3]:.2f}")

        if st.button('Use Generated Values'):
            st.session_state['input_features'] = st.session_state['generated_values']
            st.success("Generated values have been used. Navigate to the Results page to see the predictions.")

    st.markdown("**Or manually input values:**")
    sensor_1 = st.slider('Sensor 1', float(data['sensor_1'].min()), float(data['sensor_1'].max()), float(data['sensor_1'].mean()))
    sensor_2 = st.slider('Sensor 2', float(data['sensor_2'].min()), float(data['sensor_2'].max()), float(data['sensor_2'].mean()))
    sensor_3 = st.slider('Sensor 3', float(data['sensor_3'].min()), float(data['sensor_3'].max()), float(data['sensor_3'].mean()))
    operational_hours = st.slider('Operational Hours', int(data['operational_hours'].min()), int(data['operational_hours'].max()), int(data['operational_hours'].mean()))

    if st.button('Submit'):
        st.session_state['input_features'] = [sensor_1, sensor_2, sensor_3, operational_hours]
        st.success("Input data submitted successfully! Navigate to the Results page to see the predictions.")

elif selected == "Results":
    st.title("📊 Prediction Results")
    if 'input_features' not in st.session_state:
        st.warning("Please input data first in the 'Input Data' section.")
    else:
        input_features = st.session_state['input_features']
        prediction = predict_maintenance(input_features)
        st.write(f"**Remaining Useful Life (RUL):** {prediction['RUL Prediction']:.2f} hours")
        st.write(f"**Maintenance Status:** {prediction['Maintenance Prediction']}")
        st.write(f"**Anomaly Detection:** {prediction['Anomaly Detection']}")
        if prediction['Maintenance Prediction'] == 'Needs Maintenance':
            st.error('⚠️ Maintenance is required!')
        if prediction['Anomaly Detection'] == 'Anomaly':
            st.warning('⚠️ Anomaly detected in sensor readings!')

elif selected == "Visualizations":
    st.title("📊 Data Visualizations")

    # Histogram for sensor readings
    st.subheader("Histogram of Sensor Readings")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True)
    axs[0].set_title('Sensor 1')
    sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True)
    axs[1].set_title('Sensor 2')
    sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True)
    axs[2].set_title('Sensor 3')
    st.pyplot(fig)

    # Scatter plot for sensor readings vs operational hours
    st.subheader("Scatter Plot of Sensor Readings vs Operational Hours")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.5)
    axs[0].set_title('Operational Hours vs Sensor 1')
    axs[0].set_xlabel('Operational Hours')
    axs[0].set_ylabel('Sensor 1')
    axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.5)
    axs[1].set_title('Operational Hours vs Sensor 2')
    axs[1].set_xlabel('Operational Hours')
    axs[1].set_ylabel('Sensor 2')
    axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.5)
    axs[2].set_title('Operational Hours vs Sensor 3')
    axs[2].set_xlabel('Operational Hours')
    axs[2].set_ylabel('Sensor 3')
    st.pyplot(fig)

    # Line chart for RUL over time
    st.subheader("Line Chart of RUL Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-')
    ax.set_title('RUL Over Operational Hours')
    ax.set_xlabel('Operational Hours')
    ax.set_ylabel('RUL')
    st.pyplot(fig)

    if 'input_features' in st.session_state:
        input_features = st.session_state['input_features']

        # Overlay generated input values if available
        if input_features is not None:
            # Histogram for sensor readings with generated input
            st.subheader("Histogram of Sensor Readings with Generated Input")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            sns.histplot(data['sensor_1'], bins=30, ax=axs[0], kde=True)
            axs[0].set_title('Sensor 1')
            axs[0].axvline(input_features[0], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['sensor_2'], bins=30, ax=axs[1], kde=True)
            axs[1].set_title('Sensor 2')
            axs[1].axvline(input_features[1], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['sensor_3'], bins=30, ax=axs[2], kde=True)
            axs[2].set_title('Sensor 3')
            axs[2].axvline(input_features[2], color='red', linestyle='--', label='Generated Value')
            plt.legend()
            st.pyplot(fig)

            # Scatter plot for sensor readings vs operational hours with generated input
            st.subheader("Scatter Plot of Sensor Readings vs Operational Hours with Generated Input")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].scatter(data['operational_hours'], data['sensor_1'], alpha=0.5)
            axs[0].set_title('Operational Hours vs Sensor 1')
            axs[0].set_xlabel('Operational Hours')
            axs[0].set_ylabel('Sensor 1')
            axs[0].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            axs[0].legend()
            axs[1].scatter(data['operational_hours'], data['sensor_2'], alpha=0.5)
            axs[1].set_title('Operational Hours vs Sensor 2')
            axs[1].set_xlabel('Operational Hours')
            axs[1].set_ylabel('Sensor 2')
            axs[1].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            axs[1].legend()
            axs[2].scatter(data['operational_hours'], data['sensor_3'], alpha=0.5)
            axs[2].set_title('Operational Hours vs Sensor 3')
            axs[2].set_xlabel('Operational Hours')
            axs[2].set_ylabel('Sensor 3')
            axs[2].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            axs[2].legend()
            st.pyplot(fig)

            # Line chart for RUL over time with generated input
            st.subheader("Line Chart of RUL Over Time with Generated Input")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-')
            ax.set_title('RUL Over Operational Hours')
            ax.set_xlabel('Operational Hours')
            ax.set_ylabel('RUL')
            ax.axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            ax.legend()
            st.pyplot(fig)
