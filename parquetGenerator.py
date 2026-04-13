import itertools
import sys
import time
import pandas as pd
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import SOAparams
from soa_main import run_simulation  

def get_parameter_grid():
    # Los rangos definidos para SOA5
    bit_rates = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9, 11e9, 12e9, 13e9, 14e9, 15e9]
    betas_rc = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rangos_corr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    semillas = [4, 42]
    num_bits = [2**15]

    keys = ['bit_rate', 'beta_rc', 'rango_corriente', 'semilla', 'num_bits']
    combinations = list(itertools.product(bit_rates, betas_rc, rangos_corr, semillas, num_bits))
    
    grid = [dict(zip(keys, combo)) for combo in combinations]
    return grid

def run_massive_generator():
    grid = get_parameter_grid()
    total_combinations = len(grid)
    
    print(f"\n=======================================================")
    print(f"INICIANDO GENERACIÓN DE DATASET EN PARQUET (RAMA SOA5)")
    print(f"Total de escenarios a simular: {total_combinations}")
    
    out_dir = "HighQualityDataset_SOA5.parquet"
    os.makedirs(out_dir, exist_ok=True)
        
    start_time_total = time.time()
    
    for i, config in enumerate(grid):
        print(f"\n[{i+1}/{total_combinations}] Simulando: {config}")
        
        p = SOAparams()
        p.bit_rate = config['bit_rate']
        p.n_bits = config['num_bits']
        p.n_samples = int((p.n_bits / 2) * p.samples_per_bit)
        p.fs = p.samples_per_bit * p.bit_rate
        p.sample_period = 1.0 / p.fs
        
        p.beta_rc = config['beta_rc']
        p.rango_corriente = config['rango_corriente']
        p.semilla = config['semilla']

        try:
            # 1. Atrapamos TODAS las variables (ya no usamos el guion bajo para I_trim)
            I_trim, P_out_trim, sym2, P_out_trim2, time_vec2 = run_simulation(p_custom=p)
            
            # 2. Como P_out_trim2 tiene saltos de 16, le aplicamos el mismo salto a I_trim
            # para que las columnas del DataFrame tengan exactamente el mismo tamaño
            I_trim2 = I_trim[0::16]
            
            # 3. Calculamos la longitud mínima contemplando a I_trim2
            longitud_minima = min(len(time_vec2), len(sym2), len(P_out_trim2), len(I_trim2))
            
            df = pd.DataFrame({
                # Señales Nuevas de SOA5 Recortadas
                'Time': time_vec2[:longitud_minima],
                'Input_Sym2': sym2[:longitud_minima],
                'Output_P2': P_out_trim2[:longitud_minima],
                'I_trim': I_trim2[:longitud_minima], # <--- ¡Aquí está tu columna nueva!
                
                # Constantes
                'Bit_Rate': config['bit_rate'],
                'Beta_RC': config['beta_rc'],
                'Rango_Corr': config['rango_corriente'],
            })
            
            part_file = f"{out_dir}/part_{i:04d}.parquet"
            df.to_parquet(part_file, engine='pyarrow', index=False)
            
        except Exception as e:
            print(f"Error en la iteración {i}: {e}")
            
    print(f"\nTerminado en {(time.time() - start_time_total)/60:.2f} minutos.")

if __name__ == "__main__":
    run_massive_generator()