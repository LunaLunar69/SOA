import pandas as pd
import numpy as np
import glob
import os

def calculate_baseline_ber(file_path):
    # 1. Cargar datos
    df = pd.read_parquet(file_path)
    
    # 2. Normalizar ambas señales (0 a 1) para que sean comparables
    # Esto es vital porque Output_P2 suele tener valores mucho más pequeños
    y_true = df['I_trim'].values
    y_distorted = df['Output_P2'].values
    
    y_true_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 1e-7)
    y_distorted_norm = (y_distorted - y_distorted.min()) / (y_distorted.max() - y_distorted.min() + 1e-7)
    
    def get_ber(t_true, t_dist, threshold):
        bits_true = (t_true > 0.5).astype(int) # Asumimos 0.5 para la original limpia
        bits_dist = (t_dist > threshold).astype(int)
        return np.mean(bits_true != bits_dist)

    best_ber = 1.0
    best_delay = 0
    best_threshold = 0.5

    # 3. Buscar la mejor sincronización y umbral para la señal cruda
    # Probamos un rango de delay para compensar el tiempo de tránsito del SOA
    for d in range(-10, 11):
        for th in np.arange(0.3, 0.7, 0.01):
            if d > 0:
                t_true, t_dist = y_true_norm[d:], y_distorted_norm[:-d]
            elif d < 0:
                t_true, t_dist = y_true_norm[:d], y_distorted_norm[-d:]
            else:
                t_true, t_dist = y_true_norm, y_distorted_norm
            
            curr_ber = get_ber(t_true, t_dist, th)
            if curr_ber < best_ber:
                best_ber = curr_ber
                best_delay = d
                best_threshold = th

    return best_ber, best_delay, best_threshold

# --- EJECUCIÓN ---
files = glob.glob(os.path.join('datasets1', '*.parquet'))
archivo_test = files[-1]

ber_base, delay, th = calculate_baseline_ber(archivo_test)

print(f"--- COMPARATIVA DE RENDIMIENTO ---")
print(f"Archivo: {archivo_test}")
print(f"BER Original (Sin Ecualizar): {ber_base:.4e}")
print(f"Delay natural del SOA: {delay} muestras")
print(f"Umbral óptimo para señal cruda: {th:.3f}")
print(f"----------------------------------")