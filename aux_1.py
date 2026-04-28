import streamlit as st
import numpy as np
import joblib

# -------------------------------------------------------------------
# 1. LOAD ALL PRE-TRAINED ARTIFACTS
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all models, scalers, and bounds needed for live inference."""
    kmeans = joblib.load('models/kmeans_regime_classifier.joblib')
    xgb_model = joblib.load('models/champion_xgb_model.joblib')
    pca = joblib.load('models/pca_transformer.joblib')
    gmm = joblib.load('models/gmm_anomaly_detector.joblib')

    # Load the dictionary of 6 scalers for regime normalization
    regime_scalers = joblib.load('models/regime_scalers_dict.joblib')

    # Load the scaler used for the PCA model
    pca_scaler = joblib.load('models/pca_scaler.joblib')

    # Load the confidence score bounds (lower_bound, upper_bound)
    conf_bounds = joblib.load('models/confidence_bounds.joblib')

    return kmeans, xgb_model, pca, gmm, regime_scalers, pca_scaler, conf_bounds

# -------------------------------------------------------------------
# 2. THE LIVE FEATURE ENGINEERING PIPELINE (LEAK-FREE)
# -------------------------------------------------------------------
def engineer_features_live(df, window_span=15):
    """This function performs all feature engineering steps on a live data slice."""
    data = df.copy() # Use .copy() to avoid SettingWithCopyWarning

    # A. Regime Classification
    op_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    data['flight_regime'] = kmeans.predict(data[op_settings])

    # B. Leak-Free Normalization
    sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
            'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
            'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']
    normalized_cols = [f'{s}_norm' for s in sensors]

    for col in normalized_cols:
        data[col] = 0.0 # Initialize

    for regime in data['flight_regime'].unique():
        if regime in regime_scalers:
            scaler = regime_scalers[regime]
            regime_mask = data['flight_regime'] == regime
            data.loc[regime_mask, normalized_cols] = scaler.transform(data.loc[regime_mask, sensors])

    # C. EWMA Smoothing
    ewma_features = []
    for col in normalized_cols:
        data[f'{col}_ewma_mean'] = data[col].ewm(span=window_span, adjust=False).mean()
        data[f'{col}_ewma_std'] = data[col].ewm(span=window_span, adjust=False).std().fillna(0)
        ewma_features.extend([f'{col}_ewma_mean', f'{col}_ewma_std'])

    final_features = normalized_cols + ewma_features
    return data, final_features

# -------------------------------------------------------------------
# 3. THE LIVE ANOMALY DETECTION MECHANISM
# -------------------------------------------------------------------
def get_live_confidence(df_engineered, final_features):
    """Takes engineered data and returns the confidence score."""
    # A. Scale the data using the PCA scaler
    data_scaled = pca_scaler.transform(df_engineered[final_features])

    # B. Apply PCA transformation
    pca_coords = pca.transform(data_scaled)
    df_engineered['PC1'] = pca_coords[:, 0]
    df_engineered['PC2'] = pca_coords[:, 1]

    # C. Score with the GMM
    log_probs = gmm.score_samples(df_engineered[['PC1', 'PC2']])

    # D. Convert to 0-100 score using the saved bounds
    lower_bound, upper_bound = conf_bounds
    smooth_scores = np.interp(log_probs, (lower_bound, upper_bound), (1, 100))
    df_engineered['Confidence_Score'] = smooth_scores

    return df_engineered

def get_status_pill(value, threshold=1.5):
    '''Create colored status pills'''
    if abs(value) > threshold:
        color = "#FF4B4B"  # Red for Warning
        status_text = "Warning"
    else:
        color = "#2ECC71"  # Green for Normal
        status_text = "Normal"

    # Use HTML/CSS to create the pill
    pill_html = f"""
    <div style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 15px; text-align: center; font-weight: bold;">
        {status_text}
    </div>
    """
    return pill_html