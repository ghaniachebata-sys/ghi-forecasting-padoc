# ============================================================
# PURE DAY-AHEAD GHI FORECASTING WITHOUT SMOOTHING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 0. FIX RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("donnés pv.csv")
df.columns = df.columns.str.strip()
df = df.rename(columns={"GHI (W/m2)": "GHI"})
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
df = df.set_index("datetime")

data = df[["GHI"]].dropna()


# ============================================================
# 2. TIME FEATURES
# ============================================================

def add_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    tm = df["hour"] * 60 + df["minute"]

    df["time_sin"] = np.sin(2 * np.pi * tm / (24 * 60))
    df["time_cos"] = np.cos(2 * np.pi * tm / (24 * 60))

    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return df


data = add_time_features(data)

# ============================================================
# 3. LAGS (DNN)
# ============================================================

for i in range(1, 13):
    data[f"GHI_lag{i}"] = data["GHI"].shift(i)
data = data.dropna()

# ============================================================
# 4. FEATURES
# ============================================================

features_dnn = [f"GHI_lag{i}" for i in range(1, 13)] + ["time_sin", "time_cos", "doy_sin", "doy_cos"]
features_seq = ["GHI", "time_sin", "time_cos", "doy_sin", "doy_cos"]
target = "GHI"

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

train_data = data.loc["2016-03-01":"2016-06-30 23:55"]
test_data = data.loc["2016-07-01"]
y_true = test_data["GHI"].values

# ============================================================
# 6. SCALING
# ============================================================

# DNN
scaler_X_dnn = StandardScaler()
scaler_y_dnn = StandardScaler()
X_train_dnn = scaler_X_dnn.fit_transform(train_data[features_dnn])
y_train_dnn = scaler_y_dnn.fit_transform(train_data[[target]])

# Sequences
look_back = 12
nf = len(features_seq)
scaler_X_seq = StandardScaler()
scaler_y_seq = StandardScaler()
X_seq = scaler_X_seq.fit_transform(train_data[features_seq])
y_seq = scaler_y_seq.fit_transform(train_data[[target]])


# ============================================================
# 7. CREATE SEQUENCES
# ============================================================

def create_sequences(X, y, lb):
    Xs, ys = [], []
    for i in range(lb, len(X)):
        Xs.append(X[i - lb:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


X_train_seq, y_train_seq = create_sequences(X_seq, y_seq, look_back)

# ============================================================
# 8. MODELS
# ============================================================

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# ---- DNN ----
dnn = Sequential([
    Dense(64, activation="relu", input_dim=X_train_dnn.shape[1]),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1)
])
dnn.compile(optimizer="adam", loss="mse")
history_dnn = dnn.fit(X_train_dnn, y_train_dnn, epochs=200, batch_size=32,
                      validation_split=0.2, callbacks=[early_stop], verbose=0, shuffle=False)


# ---- Sequence Models ----
def build_seq_rnn_two_layers():
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(look_back, nf)),
        Dropout(0.2),
        SimpleRNN(16),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_seq_two_layers(layer_type):
    model = Sequential([
        layer_type(64, return_sequences=True, input_shape=(look_back, nf)),
        Dropout(0.2),
        layer_type(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


rnn = build_seq_rnn_two_layers()
lstm = build_seq_two_layers(LSTM)
gru = build_seq_two_layers(GRU)

history_rnn = rnn.fit(X_train_seq, y_train_seq, epochs=200, batch_size=32,
                      validation_split=0.2, callbacks=[early_stop], verbose=0, shuffle=False)

history_lstm = lstm.fit(X_train_seq, y_train_seq, epochs=200, batch_size=32,
                        validation_split=0.2, callbacks=[early_stop], verbose=0, shuffle=False)

history_gru = gru.fit(X_train_seq, y_train_seq, epochs=200, batch_size=32,
                      validation_split=0.2, callbacks=[early_stop], verbose=0, shuffle=False)


# ============================================================
# 9. TIME FEATURES FOR FORECAST
# ============================================================

def next_time_features(t):
    tm = t.hour * 60 + t.minute
    return (
        np.sin(2 * np.pi * tm / (24 * 60)),
        np.cos(2 * np.pi * tm / (24 * 60)),
        np.sin(2 * np.pi * t.dayofyear / 365),
        np.cos(2 * np.pi * t.dayofyear / 365)
    )


# ============================================================
# 10. PURE DAY-AHEAD FORECAST WITHOUT SMOOTHING
# ============================================================

def forecast_dnn(model, history, steps):
    hist = history.copy()
    preds = []

    for _ in range(steps):
        X = scaler_X_dnn.transform(hist.iloc[-1:][features_dnn])
        y_pred = scaler_y_dnn.inverse_transform(model.predict(X, verbose=0))[0, 0]
        preds.append(y_pred)

        t = hist.index[-1] + pd.Timedelta(minutes=5)
        ts, tc, ds, dc = next_time_features(t)
        new = pd.DataFrame([[y_pred, ts, tc, ds, dc]],
                           index=[t],
                           columns=["GHI", "time_sin", "time_cos", "doy_sin", "doy_cos"])
        for i in range(1, 13):
            new[f"GHI_lag{i}"] = hist.iloc[-i]["GHI"]
        hist = pd.concat([hist, new])

    return np.array(preds)


def forecast_seq(model, history, steps):
    hist = history.copy()
    preds = []

    for _ in range(steps):
        seq = scaler_X_seq.transform(hist[features_seq].iloc[-look_back:])
        y_pred = scaler_y_seq.inverse_transform(
            model.predict(seq.reshape(1, look_back, nf), verbose=0)
        )[0, 0]
        preds.append(y_pred)

        t = hist.index[-1] + pd.Timedelta(minutes=5)
        ts, tc, ds, dc = next_time_features(t)
        new = pd.DataFrame([[y_pred, ts, tc, ds, dc]],
                           index=[t],
                           columns=features_seq)
        hist = pd.concat([hist, new])

    return np.array(preds)


steps = len(test_data)
y_pred_dnn = forecast_dnn(dnn, train_data, steps)
y_pred_rnn = forecast_seq(rnn, train_data, steps)
y_pred_lstm = forecast_seq(lstm, train_data, steps)
y_pred_gru = forecast_seq(gru, train_data, steps)


# ============================================================
# 11. METRICS
# ============================================================

def metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred)
    )


print("\n===== PURE DAY-AHEAD GHI FORECAST WITHOUT SMOOTHING =====")
print("DNN :", metrics(y_true, y_pred_dnn))
print("RNN :", metrics(y_true, y_pred_rnn))
print("LSTM:", metrics(y_true, y_pred_lstm))
print("GRU :", metrics(y_true, y_pred_gru))

# ============================================================
# 12. VISUALIZATION
# ============================================================

# --- Time series ---
plt.figure(figsize=(15, 6))
plt.plot(test_data.index, y_true, 'k', lw=2.5, label='Observed')
plt.plot(test_data.index, y_pred_dnn, '--', label='DNN')
plt.plot(test_data.index, y_pred_rnn, '--', label='RNN')
plt.plot(test_data.index, y_pred_lstm, '--', label='LSTM')
plt.plot(test_data.index, y_pred_gru, '--', label='GRU')
plt.title("Day-ahead GHI prediction – DL models comparison (1 July 2016)")
plt.xlabel("Time")
plt.ylabel("GHI (W/m²)")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Scatter plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
models_pred = [("DNN", y_pred_dnn), ("RNN", y_pred_rnn), ("LSTM", y_pred_lstm), ("GRU", y_pred_gru)]

for ax, (name, y_pred) in zip(axes.flat, models_pred):
    ax.scatter(y_true, y_pred, alpha=0.4)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_title(f"{name}: Observed vs Predicted")
    ax.set_xlabel("Observed GHI (W/m²)")
    ax.set_ylabel("Predicted GHI (W/m²)")
    ax.grid(alpha=0.3)

plt.suptitle("Observed vs Predicted GHI – DL Models ", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Boxplot errors ---
errors = [y_true - y_pred_dnn, y_true - y_pred_rnn, y_true - y_pred_lstm, y_true - y_pred_gru]
plt.figure(figsize=(8, 5))
plt.boxplot(errors, labels=["DNN", "RNN", "LSTM", "GRU"], showfliers=False)
plt.axhline(0, color="gray", linestyle="--")
plt.ylabel("Prediction error (W/m²)")
plt.title("Error distribution comparison – DL models")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Optional: Training & Validation loss ---
plt.figure(figsize=(12, 5))
plt.plot(history_dnn.history['loss'], label='DNN train loss')
plt.plot(history_dnn.history['val_loss'], label='DNN val loss')
plt.plot(history_rnn.history['loss'], label='RNN train loss')
plt.plot(history_rnn.history['val_loss'], label='RNN val loss')
plt.plot(history_lstm.history['loss'], label='LSTM train loss')
plt.plot(history_lstm.history['val_loss'], label='LSTM val loss')
plt.plot(history_gru.history['loss'], label='GRU train loss')
plt.plot(history_gru.history['val_loss'], label='GRU val loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# ============================================================
# 13. IDENTIFIER LE MEILLEUR MODÈLE DL
# ============================================================

# Calcul des métriques pour tous les modèles
results_dl = {}
models_pred = {"DNN": y_pred_dnn, "RNN": y_pred_rnn, "LSTM": y_pred_lstm, "GRU": y_pred_gru}

for name, y_pred in models_pred.items():
    r2, rmse, mae = metrics(y_true, y_pred)
    results_dl[name] = {"pred": y_pred, "R2": r2, "RMSE": rmse, "MAE": mae}

# Trouver le meilleur modèle selon le RMSE (plus petit RMSE)
best_model_name = min(results_dl, key=lambda x: results_dl[x]["RMSE"])
best_model_pred = results_dl[best_model_name]["pred"]

print("\n===== MEILLEUR MODÈLE DL =====")
print(f"Nom du modèle : {best_model_name}")
print(f"R²   = {results_dl[best_model_name]['R2']:.4f}")
print(f"RMSE = {results_dl[best_model_name]['RMSE']:.2f} W/m²")
print(f"MAE  = {results_dl[best_model_name]['MAE']:.2f} W/m²")

# ============================================================
# 14. SAUVEGARDER LES PRÉDICTIONS DU MEILLEUR MODÈLE
# ============================================================

pred_df = pd.DataFrame({
    "datetime": test_data.index,
    "GHI_pred": best_model_pred
})
pred_df.to_csv(f"GHI_forecast_best_DL_{best_model_name}_2016-07-01.csv", index=False)

# ============================================================
# 15. SAUVEGARDER LES MÉTRIQUES
# ============================================================

metrics_df = pd.DataFrame({
    "Model": list(results_dl.keys()),
    "R2": [results_dl[m]["R2"] for m in results_dl],
    "RMSE": [results_dl[m]["RMSE"] for m in results_dl],
    "MAE": [results_dl[m]["MAE"] for m in results_dl]
})
metrics_df.to_csv(f"GHI_forecast_DL_metrics_2016-07-01.csv", index=False)

print("\n✅ Prédictions et métriques sauvegardées pour tous les modèles DL")
print(f"✅ Meilleur modèle : {best_model_name}, fichier : GHI_forecast_best_DL_{best_model_name}_2016-07-01.csv")
