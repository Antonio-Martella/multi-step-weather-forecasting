# Multi-Step Time Series Forecasting: A Deep Learning Approach on the Jena Climate Dataset

This project explores **Multi-Step Time Series Forecasting** applied to climate data. The goal is to predict the **Temperature (°C)** for the next **24 hours**, given a historical window of the past **120 hours** (5 days), comparing linear baselines against state-of-the-art Deep Learning architectures.


## Abstract

Forecasting weather variables is a challenging task due to the complex, non-linear, and chaotic nature of atmospheric dynamics. In this repository, we benchmark three different architectures:

1. **DLinear**: A simple yet effective decomposition-based linear model.

2. **BiLSTM**: A deep recurrent network (Bidirectional LSTM) capturing sequential dependencies.

3. **TSMixer**: A modern "All-MLP" architecture (Google Research, 2023) that achieves State-of-the-Art results by separating time-mixing and feature-mixing operations.


## Dataset
We used the **Jena Climate Dataset** recorded by the Max Planck Institute for Biogeochemistry.

* **Location**: Jena, Germany.

* **Resolution**: Original 10-minute intervals (Resampled to **Hourly** for this project).

* **Variables**: 14 features including Temperature, Pressure, Humidity, Wind Vector, etc.

* **Preprocessing**: Linear interpolation for missing values, Log-transformation for skewed features (Vapor Pressure), Cyclical encoding for time (Day/Year sin/cos).


## Methodological Pipeline

1. **Data Cleaning & Initial Processing**
    * **Duplicate Removal**: Identification and elimination of duplicate timestamps to ensure data integrity.
    * **Missing Values**: Handling missing data via linear interpolation to maintain time-series continuity.

2. **Exploratory Data Analysis (EDA)**
    * **Time Series Visualization**: Plotting variables over time to identify trends and seasonality.
    * **Wind Data Fix**: Correction of anomalies in wind velocity data (handling $-9999$ values).

3. **Feature Engineering**
    * **Feature Selection**: Dropping redundant variables `(H2OC (mmol/mol), sh (g/kg), Tpot (K), max. wv (m/s))` to reduce collinearity.
    * **Wind Vectorization**: Converting wind speed and direction into vector components: `(wv (m/s), wd (deg)) -> (wx (m/s), wy (m/s))`.
    * **Log-Transformation**: Applying log1p to skewed variables `(VPdef, VPmax, VPact)` to normalize their distributions.
    * **Cyclical Time Encoding**: Transforming timestamps into continuous sine/cosine signals (`Day sin/cos, Year sin/cos`) to capture daily and annual cycles.
    * **Correlation Analysis**: Heatmap evaluation to finalize the input feature set.

4. **Preprocessing**
    * **Downsampling**: reducing the dataset frequency from $10$ minutes to $1$ hour to reduce noise and computational load.
    * **Splitting & Scaling**: Dividing data into Training ($70%$), Validation ($20%$), and Test ($10%$) sets, followed by Standard Scaling (Z-Score normalization).
    
5. **Modeling Strategy**
    * **Windowing**: Transforming the time series into a supervised learning task.
        - **Input Width**: $120$ hours ($5$ days).
        - **Label Width**: $24$ hours ($1$ day).
        - **Shift**: $24$ hours (ensuring non-overlapping labels for rigorous validation).

    * **Callbacks**: Implementation of `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` to optimize training and prevent overfitting.
    * **Model Architectures**:
        - **DLinear**: Baseline linear decomposition model.
        - **BiLSTM**: Deep Recurrent Neural Network.
        - **TSMixer**: State-of-the-Art All-MLP architecture.

6. **Quantitative Evaluation**
    * **Metrics**: Calculation of *MAE* and *RMSE* on the denormalized Test Set.
    * **Benchmarking**: Comparison of deep models against a Naive Persistence Baseline.

7. **Graphical Inference**
    * **Visualization**: Plotting predictions vs. ground truth on randomized windows from the Test Set to qualitatively assess model performance and trend capturing.

8. **Conclusions**
   
    * General considerations on the model results and ideas on possible future developments.


## Results & Leaderboard
The models were evaluated on a held-out Test Set using **Mean Absolute Error** (MAE) and **Root Mean Squared Error** (RMSE) calculated on denormalized data (Real °C).
| Model | MAE (°C) | RMSE (°C) | Improvement vs Baseline |
| :--- | :---: | :---: | :---: |
| **TSMixer (SOTA)** | 1.66 | 2.17 | -51.4% |
| **BiLSTM** | 1.70 | 2.20 | -50.2% |
| **DLinear** | 1.90 | 2.53 | -44.4% |
| **Naive Baseline** | 3.42 | 4.58 | -- |

Key Findings
  * **Machine Learning works**: All models drastically outperformed the persistence baseline (Naive), confirming that temperature dynamics are deterministic and learnable.
  * **TSMixer prevails**: The MLP-Mixer architecture achieved the best performance (MAE ~1.66°C), proving that complex recurrent mechanisms (like LSTM) are not strictly necessary if temporal and feature mixing are handled correctly.
  * **Linearity is not enough**: While DLinear performed well, the gap (~0.25°C) compared to Deep models highlights the existence of non-linear interactions in weather data.

## Visualizations

Real vs Predicted values on 24h horizon.
![Testo Alternativo](temperature_forecast.png)

## Installation & Usage

  1. **Clone the repository:**
      ```bash
      git clone https://github.com/Antonio-Martella/multi-step-weather-forecasting.git
      cd multi-step-weather-forecasting
      ```
  2. **Install requirements:**
      ```bash
      pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
      ```
  3. **Run the Notebook**: Open `Weather_Forecasting_Project.ipynb` in Jupyter or Google Colab to reproduce the training and evaluation pipeline.

## References
  * **TSMixer Paper**: TSMixer: An All-MLP Architecture for Time Series Forecasting ([Google Research, 2023](https://arxiv.org/pdf/2303.06053)).
  * **DLinear Paper**: Are Transformers Effective for Time Series Forecasting? ([Zeng et al., 2022](https://arxiv.org/pdf/2205.13504)).
  * **Dataset**: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/).

## Author
**Antonio Martella**
*Project developed for personal portfolio.*
