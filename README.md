# GHI Forecasting using Deep Learning

## Description

This repository implements deep learning models for day-ahead forecasting of Global Horizontal Irradiance (GHI) using time series data.

The models are trained on historical photovoltaic measurements and aim to predict future solar irradiance values for energy forecasting applications.

## Models Implemented

- Deep Neural Network (DNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## Dataset

The dataset consists of real photovoltaic measurements with a 5-minute time resolution.

Main features:
- Global Horizontal Irradiance (GHI)
- Time-based features (sin/cos encoding)
- Lag features (1 to 12 previous values)

## Methodology

- Data preprocessing and cleaning  
- Feature engineering (time + lag features)  
- Standardization of inputs  
- Train/test split based on time series  
- Recursive forecasting strategy  
- Model evaluation using regression metrics  

## Training

Models are trained using TensorFlow/Keras with:

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Early stopping to prevent overfitting
- Batch size: 32
- Epochs: up to 200

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

```bash
python main.py
