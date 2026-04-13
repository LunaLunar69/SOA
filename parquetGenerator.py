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
    keys = ['bit_rate', 'beta_rc', 'rango_corriente', 'semilla', 'num_bits']
    num_bits = [2**15] # Fijo para todos

    # ---------------------------------------------------------
    # 1. CASOS FEOS (500 casos) -> Altas velocidades, poca corriente
    # Matemáticas: 5 tasas * 2 corrientes * 5 betas * 10 semillas = 500
    # ---------------------------------------------------------
    feos_rates = [11e9, 12e9, 13e9, 14e9, 15e9]
    feos_corr = [0.2, 0.3]
    feos_beta = [0.4, 0.5, 0.6, 0.7, 0.8]
    feos_seeds = list(range(10)) # Semillas del 0 al 9
    feos_combo = list(itertools.product(feos_rates, feos_beta, feos_corr, feos_seeds, num_bits))

    # ---------------------------------------------------------
    # 2. CASOS BONITOS (250 casos) -> Bajas velocidades, mucha corriente
    # Matemáticas: 5 tasas * 2 corrientes * 5 betas * 5 semillas = 250
    # ---------------------------------------------------------
    bonitos_rates = [1e9, 2e9, 3e9, 4e9, 5e9]
    bonitos_corr = [0.6, 0.7]
    bonitos_beta = [0.4, 0.5, 0.6, 0.7, 0.8]
    bonitos_seeds = list(range(10, 15)) # Semillas del 10 al 14
    bonitos_combo = list(itertools.product(bonitos_rates, bonitos_beta, bonitos_corr, bonitos_seeds, num_bits))

    # ---------------------------------------------------------
    # 3. CASOS GENERALES / MID (250 casos) -> Velocidades y corrientes medias
    # Matemáticas: 5 tasas * 2 corrientes * 5 betas * 5 semillas = 250
    # ---------------------------------------------------------
    mid_rates = [6e9, 7e9, 8e9, 9e9, 10e9]
    mid_corr = [0.4, 0.5]
    mid_beta = [0.4, 0.5, 0.6, 0.7, 0.8]
    mid_seeds = list(range(15, 20)) # Semillas del 15 al 19
    mid_combo = list(itertools.product(mid_rates, mid_beta, mid_corr, mid_seeds, num_bits))

    # Juntamos los 3 escuadrones
    todas_las_combinaciones = feos_combo + bonitos_combo + mid_combo
    
    # Convertimos la lista de tuplas a la lista de diccionarios que espera tu código
    grid = [dict(zip(keys, combo)) for combo in todas_las_combinaciones]
    return grid

def run_massive_generator():
    grid = get_parameter_grid()
    total_combinations = len(grid)
    
    print(f"\n=======================================================")
    print(f"🔥 INICIANDO GENERACIÓN DE DATASET EN PARQUET (RAMA SOA5) 🔥")
    print(f"Total de escenarios a simular: {total_combinations} (500 Feos, 250 Bonitos, 250 Mid)")
    
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
            I_trim, P_out_trim, sym2, P_out_trim2, time_vec2 = run_simulation(p_custom=p)
            
            I_trim2 = I_trim[0::16]
            
            longitud_minima = min(len(time_vec2), len(sym2), len(P_out_trim2), len(I_trim2))
            
            df = pd.DataFrame({
                # FIX: Borré la línea duplicada de 'Time' que tenías en tu código original
                'Time': time_vec2[:longitud_minima],
                'Input_Sym2': sym2[:longitud_minima],
                'Output_P2': P_out_trim2[:longitud_minima],
                'I_trim': I_trim2[:longitud_minima], 
                
                # Constantes
                'Bit_Rate': config['bit_rate'],
                'Beta_RC': config['beta_rc'],
                'Rango_Corr': config['rango_corriente'],
            })
            
            part_file = f"{out_dir}/part_{i:04d}.parquet"
            df.to_parquet(part_file, engine='pyarrow', index=False)
            
        except Exception as e:
            print(f"❌ Error en la iteración {i}: {e}")
            
    print(f"\n✅ Terminado en {(time.time() - start_time_total)/60:.2f} minutos.")

if __name__ == "__main__":
    run_massive_generator()