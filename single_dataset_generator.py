import itertools
import time
import pandas as pd
import os
from params import SOAparams
from soa_main import run_simulation  

def get_parameter_grid():
    bit_rates = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9, 11e9, 12e9, 13e9, 14e9, 15e9, 16e9, 17e9, 18e9, 19e9, 20e9, 21e9, 22e9, 23e9, 24e9, 25e9]
    betas_rc = [0.1, 0.3, 0.5, 0.8, 1.0]
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
    print(f"INICIANDO GENERACIÓN MASIVA DE DATASET")
    print(f"Total de escenarios a simular: {total_combinations}")
    
    # Crear carpeta de salida
    out_dir = "datasets_results/single_dataset"
    os.makedirs(out_dir, exist_ok=True)
    output_file = f"{out_dir}/HighQualityDataset.csv"
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    start_time_total = time.time()
    
    for i, config in enumerate(grid):
        print(f"[{i+1}/{total_combinations}] Simulando: {config}")
        
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
            I_trim, P_out_trim = run_simulation(p)[:2]
            
            # 3. Guardar en CSV 
            df = pd.DataFrame({
                'Current_Input_A': I_trim,
                'Power_Output_mW': P_out_trim * 1e3
            })
            
            # asegura que los títulos solo se pongan 1 vez hasta arriba.
            # para cada iteración se añade al final del CSV sin sobreescribir lo anterior.
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            
        except Exception as e:
            print(f"Error en la iteración {i}: {e}")
            
    print(f"\nTerminado en {(time.time() - start_time_total)/60:.2f} minutos.")

if __name__ == "__main__":
    run_massive_generator()