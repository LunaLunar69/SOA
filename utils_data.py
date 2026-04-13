import pandas as pd
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
import os
import glob 
import json
from numpy.lib.stride_tricks import sliding_window_view

FEATURES = ['Output_P2', 'Bit_Rate', 'Beta_RC', 'Rango_Corr']
TARGET = ['I_trim']
ALL_COLUMNS = FEATURES + TARGET

def calculate_global_scalers(file_list, chunksize=1000000, save_path='scalers.json'):
    print(f"Buscando y calculando scalers sobre {len(file_list)} archivos...")
    
    if not file_list:
        raise ValueError("No se encontraron archivos .parquet en la lista proporcionada.")

    mins = {col: float('inf') for col in ALL_COLUMNS}
    maxs = {col: float('-inf') for col in ALL_COLUMNS}
    
    for file_path in file_list:
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
    def __init__(self, file_list, window_size, batch_size=512, scalers_path='scalers.json'):
        self.file_list = file_list 
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_features = len(FEATURES)
        
        if not os.path.exists(scalers_path):
            print("No se encontró el archivo de scalers. Iniciando cálculo masivo...")
            calculate_global_scalers(self.file_list, save_path=scalers_path)
            
        with open(scalers_path, 'r') as f:
            self.scalers = json.load(f)
            
    def _normalize(self, data, col_name):
        v_min = self.scalers['min'][col_name]
        v_max = self.scalers['max'][col_name]
        return (data - v_min) / (v_max - v_min + 1e-7)

    def data_generator(self):
        while True:
            num_files_to_mix = min(8, len(self.file_list))
            batch_files = np.random.choice(self.file_list, size=num_files_to_mix, replace=False)
            
            x_chunks, y_chunks = [], []
            
            for file_path in batch_files:
                try:
                    df = pd.read_parquet(file_path)
                    
                    chunk_size = 10000 
                    max_start = len(df) - self.window_size - chunk_size
                    
                    if max_start <= 0: continue
                    
                    start_idx = np.random.randint(0, max_start)
                    end_idx = start_idx + chunk_size + self.window_size
                    
                    df_chunk = df.iloc[start_idx:end_idx]
                    
                    x_scaled = np.column_stack([
                        self._normalize(df_chunk[col].values.astype(np.float32), col) 
                        for col in FEATURES
                    ])
                    y_scaled = self._normalize(df_chunk[TARGET[0]].values.astype(np.float32), TARGET[0])
                    
                    x_windows = sliding_window_view(x_scaled[:-1], (self.window_size, self.num_features)).reshape(-1, self.window_size, self.num_features)
                    
                    y_targets = y_scaled[self.window_size // 2 : self.window_size // 2 + len(x_windows)]
                    
                    x_chunks.append(x_windows)
                    y_chunks.append(y_targets)
                    
                except Exception as e:
                    print(f"Error procesando {file_path}: {e}")
                    continue
            
            if not x_chunks: continue
            
            X_batch_mixed = np.concatenate(x_chunks)
            Y_batch_mixed = np.concatenate(y_chunks)
            
            idx = np.random.permutation(len(X_batch_mixed))
            X_batch_mixed = X_batch_mixed[idx]
            Y_batch_mixed = Y_batch_mixed[idx]
            
            for i in range(0, len(X_batch_mixed), self.batch_size):
                if i + self.batch_size <= len(X_batch_mixed):
                    yield X_batch_mixed[i:i + self.batch_size], Y_batch_mixed[i:i + self.batch_size]

    def get_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(None, self.window_size, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
        
        ds = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=output_signature
        )
        
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds