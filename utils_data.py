import pandas as pd
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
import os
import glob
import json

FEATURES = ['Power_Output_mW', 'Bit_Rate', 'Beta_RC', 'Rango_Corr']
TARGET = ['Current_Input_A']
ALL_COLUMNS = FEATURES + TARGET

def calculate_global_scalers(data_dir, chunksize=1000000, save_path='scalers.json'):
    print(f"Buscando archivos en la carpeta: {data_dir}")
    # Busca todos los archivos .parquet dentro de la carpeta
    parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    
    if not parquet_files:
        raise ValueError(f"No se encontraron archivos .parquet en {data_dir}")

    mins = {col: float('inf') for col in ALL_COLUMNS}
    maxs = {col: float('-inf') for col in ALL_COLUMNS}
    
    # Iteramos sobre cada archivo encontrado
    for file_path in parquet_files:
        print(f"Calculando scalers en: {os.path.basename(file_path)}")
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=chunksize, columns=ALL_COLUMNS):
            chunk = batch.to_pandas()
            for col in ALL_COLUMNS:
                col_min = chunk[col].min()
                col_max = chunk[col].max()
                if col_min < mins[col]: mins[col] = float(col_min)
                if col_max > maxs[col]: maxs[col] = float(col_max)
            
    scalers = {'min': mins, 'max': maxs}
    
    with open(save_path, 'w') as f:
        json.dump(scalers, f, indent=4)
        
    print(f"¡Scalers calculados globalmente y guardados en {save_path}!")
    return scalers

class SOADataLoader:
    def __init__(self, data_dir, window_size, batch_size=512, scalers_path='scalers.json'):
        self.data_dir = data_dir # Ahora recibe una carpeta, no un archivo
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_features = len(FEATURES)
        
        if not os.path.exists(scalers_path):
            print("No se encontró el archivo de scalers. Iniciando cálculo masivo...")
            calculate_global_scalers(self.data_dir, save_path=scalers_path)
            
        with open(scalers_path, 'r') as f:
            self.scalers = json.load(f)
            
    def _normalize(self, data, col_name):
        v_min = self.scalers['min'][col_name]
        v_max = self.scalers['max'][col_name]
        return (data - v_min) / (v_max - v_min + 1e-7)

    def get_generator(self):
        def generator():
            # Volvemos a buscar los archivos para el entrenamiento
            parquet_files = glob.glob(os.path.join(self.data_dir, '*.parquet'))
            
            # Leemos archivo por archivo
            for file_path in parquet_files:
                parquet_file = pq.ParquetFile(file_path)
                
                for batch in parquet_file.iter_batches(batch_size=1000000, columns=ALL_COLUMNS):
                    chunk = batch.to_pandas()
                    
                    x_dict = {col: self._normalize(chunk[col].values.astype(np.float32), col) for col in FEATURES}
                    y_scaled = self._normalize(chunk[TARGET[0]].values.astype(np.float32), TARGET[0])
                    
                    x_scaled = np.column_stack([x_dict[col] for col in FEATURES])
                    
                    for i in range(len(x_scaled) - self.window_size):
                        yield x_scaled[i : i + self.window_size], y_scaled[i + self.window_size // 2]
                        
        return generator

    def get_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(self.window_size, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        
        ds = tf.data.Dataset.from_generator(
            self.get_generator(),
            output_signature=output_signature
        )
        
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds