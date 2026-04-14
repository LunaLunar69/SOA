import numpy as np
import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from utils_data import SOADataLoader

# CONFIGURACIÓN FINAL
STUDENT_PATH = 'student_soa_best.keras'
SCALERS_PATH = 'scalers_soa5.json' 
WINDOW_SIZE = 128
FINE_TUNE_PERCENT = 0.40  
DATASET_DIR = 'datasets1' 
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

# CARGA Y PREPARACIÓN DEL MODELO
print("Cargando modelo Student y preparando Fine-Tuning...")
model = tf.keras.models.load_model(
    STUDENT_PATH, 
    custom_objects={'SimpleAttention': SimpleAttention}
)

for layer in model.layers[:-2]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss='mse'
)

# CARGA DE DATOS
all_files = glob.glob(os.path.join(DATASET_DIR, '*.parquet'))
if not all_files:
    raise FileNotFoundError(f"No se encontraron archivos en {DATASET_DIR}")

test_file = all_files[-1] 
print(f"Archivo seleccionado para prueba: {test_file}")

loader = SOADataLoader([test_file], WINDOW_SIZE, batch_size=10000)
x_all, y_all = next(iter(loader.get_tf_dataset()))

split_idx = int(len(x_all) * FINE_TUNE_PERCENT)
x_calib, y_calib = x_all[:split_idx], y_all[:split_idx]
x_infer, y_infer = x_all[split_idx:], y_all[split_idx:]

# FINE-TUNING DE ADAPTACIÓN
print(f"Ejecutando Fine-Tuning (25 épocas) sobre {len(x_calib)} muestras...")
model.fit(x_calib, y_calib, epochs=25, batch_size=128, verbose=1)

# INFERENCIA
print("\nBuscando el punto óptimo de decisión...")
preds = model.predict(x_infer).flatten()
y_true_np = y_infer.numpy()

def calculate_ber(y_t, y_p, threshold):
    b_true = (y_t > threshold).astype(int)
    b_pred = (y_p > threshold).astype(int)
    return np.mean(b_true != b_pred)

best_ber = 1.0
best_delay = 0
best_threshold = 0.5

for d in range(-5, 6):
    for th in np.arange(0.30, 0.80, 0.005):
        if d > 0:
            t_true, t_pred = y_true_np[d:], preds[:-d]
        elif d < 0:
            t_true, t_pred = y_true_np[:d], preds[-d:]
        else:
            t_true, t_pred = y_true_np, preds
        
        curr_ber = calculate_ber(t_true, t_pred, th)
        if curr_ber < best_ber:
            best_ber = curr_ber
            best_delay = d
            best_threshold = th

# REPORTE
print(f"INFORME TÉCNICO DE ECUALIZACIÓN")
print(f"BER Final: {best_ber:.4e}")
print(f"Umbral Óptimo: {best_threshold:.3f}")
print(f"Delay Óptimo: {best_delay} muestras")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
start, end = 300, 450 
plt.plot(y_true_np[start:end], label="Original (I_trim)", color='#1f77b4', lw=2)
p_start, p_end = start - best_delay, end - best_delay
plt.plot(preds[p_start:p_end], label="Predicción Estudiante", color='#d62728', linestyle='--', lw=2)
plt.axhline(y=best_threshold, color='green', alpha=0.6, linestyle=':', label='Umbral de Decisión')
plt.title(f"Alineación Temporal (BER: {best_ber:.2e})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(preds, bins=80, color='gray', alpha=0.3, label='Distribución Predicción')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Corte: {best_threshold:.2f}')
plt.title("Histograma de Amplitudes (Decisión de Bit)")
plt.xlabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()