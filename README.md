# AgriPredict: ML-Driven Insights for Data-Driven Crop Production, Yield Forecasting, and Risk Mitigation

![Project Status](https://img.shields.io/badge/Status-In%20Progress-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)
- [Modeling (Time Series - SARIMA)](#modeling-time-series---sarima)
- [API Endpoints](#api-endpoints)
- [User Interface (Streamlit)](#user-interface-streamlit)
- [Dashboard (Tableau)](#dashboard-tableau)
- [Installation & Setup](#installation--setup)
- [How to Run Locally](#how-to-run-locally)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Project Overview
This project aims to forecast crop production in India leveraging historical agricultural data. It encompasses a complete data pipeline from raw data processing using big data tools to time series modeling, serving predictions via an API, and visualizing results through an interactive dashboard and UI.

## Tech Stack
Our project utilizes a robust set of technologies:

* **Programming Languages:** Python
* **Machine Learning:** Time Series Forecasting (SARIMA), General ML concepts
* **Database:** SQL (for potential future data storage/management)
* **Big Data Tools:** Hadoop (HDFS, Hive), PySpark (for large-scale data processing)
* **UI/Web Framework:** Streamlit
* **Dashboarding & Visualization:** Tableau

## Dataset
-   **Source:** [Kaggle: Crop Production in India](https://www.kaggle.com/datasets/asishpandey/crop-production-in-india)
-   **File Used:** `Data_after_rainfall.csv`
-   **Description:** The dataset contains historical crop production data across various crops, seasons, and states in India, along with supplementary information like rainfall, temperature, and soil nutrient levels (Nitrogen, Phosphorus, Potassium, pH).
-   **Data Acquisition:** The raw `Data_after_rainfall.csv` is sourced from Kaggle and placed into the `data/raw/` directory.

## Data Acquisition & Preprocessing
-   **Key Tools:** Python (Pandas), PySpark, Hadoop (HDFS, Hive for large scale)
-   **Notebook/Scripts:** `notebooks/data_preprocessing.ipynb`, potentially `src/data_processor.py`
-   **Description:** This phase involves:
    -   Loading raw data.
    -   Handling missing values, data type conversions, and outlier detection.
    -   Aggregating data to create appropriate time series (e.g., total annual production, or production per specific crop/state/season).
    -   Leveraging **PySpark** for efficient processing of large datasets, if the scale warrants it, potentially integrating with **HDFS** for distributed storage and **Hive** for data warehousing.
-   **Output:** `data/processed/final_agripredict_processed_data.csv` (or stored in Hive/HDFS for larger scale deployments)

## Modeling (Time Series - SARIMA)
-   **Key Tools:** Python (Statsmodels for SARIMA, Scikit-learn for evaluation)
-   **Notebook/Scripts:** `notebooks/model_development.ipynb`, potentially `src/model_trainer.py`
-   **Model Used:** SARIMA (Seasonal Autoregressive Integrated Moving Average) for time series forecasting.
-   **Process:**
    -   Exploratory Data Analysis (EDA) tailored for time series, including ACF/PACF plots to identify seasonality and trend.
    -   Splitting data into training and testing sets.
    -   Training the SARIMA model with optimized parameters (p,d,q)(P,D,Q,s) for different forecasting granularities (e.g., total production, crop-specific, state-specific).
    -   Evaluating model performance using metrics like RMSE and MAE.
-   **Saved Model:** `models/sarima_model.pkl`

## API Endpoints
-   **Key Tool:** Python (Flask or FastAPI)
-   **Script:** `src/api.py`
-   **Description:** A RESTful API serves forecasts from the trained SARIMA model.
-   **Main Endpoint:** `/predict`
    -   **Method:** `GET` (or `POST` for more complex requests)
    -   **Parameters:** `year` (integer, required for forecast horizon), `crop` (string, optional), `state` (string, optional).
    -   **Example Request:** `GET /predict?year=2025&crop=Rice`
    -   **Example Response:**
        ```json
        {
          "year": 2025,
          "forecasted_production": 12345.67
        }
        ```

## User Interface (Streamlit)
-   **Key Tool:** Streamlit
-   **Script:** `app.py` (or similar in `ui/` directory)
-   **Description:** An interactive web application built with **Streamlit** that allows users to:
    -   Select forecasting parameters (e.g., target year, specific crop, specific state).
    -   Trigger the API call to get predictions.
    -   Visualize historical data and the forecasted values.
    -   Provides an intuitive way to interact with the forecasting model without needing to write code.

## Dashboard (Tableau)
-   **Key Tool:** Tableau
-   **File:** `dashboards/crop_production_dashboard.twbx` (or similar)
-   **Description:** A comprehensive interactive dashboard created using **Tableau** to visualize:
    -   Historical crop production trends.
    -   Comparisons across different crops, seasons, and states.
    -   Key insights from the dataset.
    -   Potentially integrate forecasted values for a holistic view of past and future production.

## Installation & Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: If using PySpark, ensure Java and Spark are configured on your system or use a Dockerized environment.)

## How to Run Locally
1.  **Prepare Data:**
    * Place `Data_after_rainfall.csv` in `data/raw/`.
    * Run `notebooks/data_preprocessing.ipynb` or `python src/data_processor.py` to generate `data/processed/final_agripredict_processed_data.csv`.
2.  **Train Model:**
    * Run `notebooks/model_development.ipynb` or `python src/model_trainer.py` to train and save `models/sarima_model.pkl`.
3.  **Start API Server:**
    ```bash
    python src/api.py
    ```
    (The API will typically run on `http://127.0.0.1:5000` or similar.)
4.  **Run Streamlit UI:**
    ```bash
    streamlit run app.py
    ```
    (This will open the UI in your web browser.)
5.  **Open Tableau Dashboard:**
    * Open `dashboards/crop_production_dashboard.twbx` using Tableau Desktop.

## Usage
-   Interact with the Streamlit UI to input parameters and view forecasts.
-   Explore historical data and trends using the Tableau dashboard.
-   Directly call the API endpoint for programmatic access to forecasts.

## Results & Evaluation
Summarize the performance of the SARIMA model (e.g., RMSE, MAE on the test set). Discuss the strengths and limitations of the current forecasting approach. Include insights from your Tableau dashboard.

## Future Work
-   Explore advanced **Machine Learning** and **Deep Learning** models for time series (e.g., LSTMs, Transformers) for potentially higher accuracy, especially if more complex patterns or exogenous variables are to be incorporated.
-   Integrate **SQL** database for more robust data storage and management, instead of flat CSV files.
-   Expand **PySpark** and **Hadoop (HDFS, Hive)** integration for truly big data scenarios and distributed model training.
-   Refine SARIMA parameters using more sophisticated auto-ARIMA or grid search techniques.
-   Add more detailed visualizations to the Streamlit app.
-   Deploy the entire application (API and Streamlit) to a cloud platform (e.g., AWS, GCP, Azure).

## Contributors
-   [Your Name/GitHub Profile Link] (e.g., Data Lead, UI Developer - Streamlit)
-   [Collaborator 2 Name/GitHub Profile Link] (e.g., Model Lead - SARIMA, Documentation)
-   [Collaborator 3 Name/GitHub Profile Link] (e.g., API Developer, Big Data Integration - PySpark/Hadoop)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.