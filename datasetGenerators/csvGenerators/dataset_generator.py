import itertools
import math
import sys
import time
import pandas as pd
import argparse
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import SOAparams
from soa_main import run_simulation  

def get_parameter_grid():
    # Los rangos definidos
    bit_rates = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9, 11e9, 12e9, 13e9, 14e9, 15e9, 16e9, 17e9, 18e9, 19e9, 20e9, 21e9, 22e9, 23e9, 24e9, 25e9]
    betas_rc = [0.1, 0.3, 0.5, 0.8, 1.0]
    rangos_corr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    semillas = [4, 42]
    num_bits = [2**15]

    # Generamos todas las combinaciones posibles
    keys = ['bit_rate', 'beta_rc', 'rango_corriente', 'semilla', 'num_bits']
    combinations = list(itertools.product(bit_rates, betas_rc, rangos_corr, semillas, num_bits))
    
    # Lo convertimos a una lista de diccionarios para que sea más fácil de leer
    grid = [dict(zip(keys, combo)) for combo in combinations]
    return grid

def run_worker(worker_id, total_workers):
    grid = get_parameter_grid()
    total_combinations = len(grid)
    
    print(f"\n=======================================================")
    print(f"Total de escenarios posibles: {total_combinations}")
    
    # --- LÓGICA DE DIVISIÓN DE TRABAJO ---
    # Calculamos cuántos escenarios le tocan a cada computadora
    chunk_size = math.ceil(total_combinations / total_workers)
    
    start_idx = worker_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_combinations)
    
    my_chunk = grid[start_idx:end_idx]
    
    print(f"PC No: {worker_id} de {total_workers-1} asignada al bloque: {start_idx} al {end_idx-1}")
    print(f"Escenarios a simular en esta máquina: {len(my_chunk)}\n")
    
    # Directorio de salida específico para esta computadora
    out_dir = f"datasets_results/dataset_worker_{worker_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- CICLO DE SIMULACIÓN ---
    start_time_total = time.time()
    
    for i, config in enumerate(my_chunk):
        print(f"\n[{i+1}/{len(my_chunk)}] Simulando: {config}")
        
        # 1. Configurar los parámetros de esta iteración
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
            # 2. Correr la simulación
            # Solamente obtenemos las señales de interés para el dataset
            # los dos primeros elementos de la tupla que devuelve run_simulation
            I_trim, P_out_trim = run_simulation(p)[:2]
            
            # 3. Guardar en CSV
            df = pd.DataFrame({
                'Current_Input_A': I_trim,
                'Power_Output_mW': P_out_trim * 1e3,
                'Bit_Rate': config['bit_rate'],
                'Beta_RC': config['beta_rc'],
                'Rango_Corr': config['rango_corriente'],
            })
            
            # Nombre de archivo dinámico
            filename = f"{out_dir}/data_br{int(config['bit_rate']/1e9)}beta_rc{config['beta_rc']}_rc{config['rango_corriente']}_seed{config['semilla']}.csv"
            df.to_csv(filename, index=False)
            
        except Exception as e:
            print(f"Error en la iteración {i}: {e}")
            
    print(f"\nTerminado en {(time.time() - start_time_total)/60:.2f} minutos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador Distribuido de Datasets SOA')
    parser.add_argument('--worker_id', type=int, required=True, help='ID de esta computadora 0, 2, 3 ...)')
    parser.add_argument('--total_workers', type=int, required=True, help='Cuántas computadoras hay en total (ej. 4)')
    args = parser.parse_args()
    
    run_worker(args.worker_id, args.total_workers)