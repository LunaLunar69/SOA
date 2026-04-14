import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from utils_data import SOADataLoader

# CONFIGURACIÓN
STUDENT_PATH = 'student_soa_best.keras'
DATASET_DIR = 'datasets1' 
WINDOW_SIZE = 128
FINE_TUNE_PERCENT = 0.40  
all_files = glob.glob(os.path.join(DATASET_DIR, '*.parquet'))
test_file = all_files[-1] 

# CAPA PERSONALIZADA
@tf.keras.utils.register_keras_serializable()
class SimpleAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.W = layers.Dense(self.units, activation='tanh')
        self.V = layers.Dense(1)
        super(SimpleAttention, self).build(input_shape)
    def call(self, inputs):
        score = self.V(self.W(inputs))
        attention_weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(attention_weights * inputs, axis=1)
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# CÁLCULO DE BER BASE
print(f"Analizando señal cruda de: {test_file}")
df_raw = pd.read_parquet(test_file)
y_true_raw = df_raw['I_trim'].values
y_soa_raw = df_raw['Output_P2'].values

# Normalización
y_t_n = (y_true_raw - y_true_raw.min()) / (y_true_raw.max() - y_true_raw.min() + 1e-7)
y_s_n = (y_soa_raw - y_soa_raw.min()) / (y_soa_raw.max() - y_soa_raw.min() + 1e-7)

ber_base, base_delay, base_th = 1.0, 0, 0.5
for d in range(-10, 11):
    for th in np.arange(0.3, 0.7, 0.01):
        if d > 0: t, p = y_t_n[d:], y_s_n[:-d]
        elif d < 0: t, p = y_t_n[:d], y_s_n[-d:]
        else: t, p = y_t_n, y_s_n
        curr = np.mean((t > 0.5) != (p > th))
        if curr < ber_base:
            ber_base, base_delay, base_th = curr, d, th

# MODELO ESTUDIANTE
print("\nCargando y entrenando Modelo Estudiante...")
model = tf.keras.models.load_model(STUDENT_PATH, custom_objects={'SimpleAttention': SimpleAttention})

for layer in model.layers[:-2]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

loader = SOADataLoader([test_file], WINDOW_SIZE, batch_size=10000)
x_all, y_all = next(iter(loader.get_tf_dataset()))

split_idx = int(len(x_all) * FINE_TUNE_PERCENT)
x_calib, y_calib = x_all[:split_idx], y_all[:split_idx]
x_infer, y_infer = x_all[split_idx:], y_all[split_idx:]

model.fit(x_calib, y_calib, epochs=25, batch_size=128, verbose=1)

# Inferencia
preds = model.predict(x_infer).flatten()
y_true_np = y_infer.numpy()

best_ber, best_delay, best_threshold = 1.0, 0, 0.5

for d in range(-5, 6):
    for th in np.arange(0.30, 0.80, 0.005):
        if d > 0: t_true, t_pred = y_true_np[d:], preds[:-d]
        elif d < 0: t_true, t_pred = y_true_np[:d], preds[-d:]
        else: t_true, t_pred = y_true_np, preds
        
        curr_ber = np.mean((t_true > th) != (t_pred > th))
        if curr_ber < best_ber:
            best_ber, best_delay, best_threshold = curr_ber, d, th

# REPORTE
print("\n" + "="*50)
print(f"RESUMEN DE MEJORA")
print("-" * 50)
print(f"BER Original:  {ber_base:.4e}")
print(f"BER Final:      {best_ber:.4e}")

if best_ber > 0:
    improvement = ber_base / best_ber
    print(f"Factor de Mejora:    {improvement:.2f}x")
else:
    print(f"Factor de Mejora:    INFINITA (Error-Free)")
print("="*50)

# GRÁFICAS
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
start, end = 300, 450
plt.plot(y_true_np[start:end], label="Target Original", color='#1f77b4', lw=2, alpha=0.6)
p_s, p_e = start - best_delay, end - best_delay
plt.plot(preds[p_s:p_e], label="Predicción IA", color='#d62728', linestyle='--')
plt.axhline(y=best_threshold, color='green', linestyle=':', label=f'Threshold: {best_threshold:.2f}')
plt.title(f"Alineación IA (BER: {best_ber:.2e})")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(preds, bins=80, color='gray', alpha=0.3)
plt.axvline(x=best_threshold, color='red', linestyle='--')
plt.title("Distribución de Amplitudes Final")
plt.show()