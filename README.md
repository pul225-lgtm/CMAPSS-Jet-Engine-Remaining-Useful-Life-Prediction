## Project Description
Unplanned jet engine failure is a catastrophic safety and financial risk. While machine learning can predict the Remaining Useful Life (RUL) of engines, traditional models fail silently when environmental operating conditions change (Covariate Shift). This project develops a "Self-Healing" MLOps pipeline. It features an Operating Regime Normalizer and Exponentially Weighted Moving Averages (EWMA) to denoise telemetry. It utilizes a tuned XGBoost regressor optimized for the NASA Asymmetric Score (penalizing late warnings) and an unsupervised Gaussian Mixture Model (GMM) on a PCA manifold to detect live data drift, deployed via an interactive Streamlit application.

## Data Source
The project utilizes the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. It consists of simulated run-to-failure turbofan engine degradation under varying flight regimes (altitude, Mach number, throttle).
Source: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## Packages Required
To run this project, ensure you have Python 3.9+ and the following packages installed:
* pandas
* numpy
* scikit-learn
* xgboost
* streamlit
* plotly
* matplotlib
* seaborn
* joblib

You can install these via: `pip install -r requirements.txt`

## Instructions to Run the Code
1. Clone this repository to your local machine.
2. Download the CMAPSS dataset and place the `.txt` files into the `/data` directory (see `/data/readme_data.txt` for details).
3. (Optional) Run `aux_1.py` (or the Jupyter Notebook) to train the models and export the `.joblib` artifacts to the `/models` folder.
4. Run the Streamlit web application by executing the following command in your terminal:
   `streamlit run main.py`
5. Open the provided `localhost` URL in your web browser to interact with the dashboard.
