import numpy as np
import time
import pandas as pd
import os
import itertools
from scipy.signal import correlate, convolve, lfilter
from numba import njit
from cw_laser import cw_laser # Este sí lo importamos de tu archivo externo

# =====================================================================
# 1. CLASE DE PARÁMETROS (Antes en params.py)
# =====================================================================
class SOAparams:
    def __init__(self):
        # Geometrical parameters
        self.L = 2.0E-3
        self.width = 2.8E-6
        self.depth = 0.25E-6
        self.Gamma = 0.4
        self.vg = 8.5E+7
        
        # Cálculos derivados
        self.Vol = self.width * self.depth * self.L
        self.Aeff = (self.width * self.depth) / self.Gamma

        # Material Parameters
        self.N0 = 0.46E24
        self.loss = 1000.0
        self.DiffGain = 5.3E-20
        self.A = 6.0E8
        self.B = 18.0E-16
        self.C = 1.0E-40
        self.LEF = 3.0 
        self.lambda_val = 1550E-9

        # Signal Parameters Base
        self.Numsymrrc = 8
        self.bit_rate = 5.35e9          
        self.n_bits = 2**14             
        self.samples_per_bit = 32
        
        # Atributos dinámicos (Se llenan en el generador)
        self.beta_rc = 0.3
        self.rango_corriente = 0.4
        self.semilla = 42

        self.n_samples = int((self.n_bits / 2) * self.samples_per_bit)
        self.fs = self.samples_per_bit * self.bit_rate
        self.sample_period = 1.0 / self.fs

# =====================================================================
# 2. MOTOR FÍSICO Y MATEMÁTICO (Antes en soa_main.py)
# =====================================================================
@njit
def SOA_N_RK4_uniform(TimeVec, I, Pseg, Ninit, dt, q, Ener, Vol_seg, Vol_total, Gamma, N0, DiffGain, A, B, C, segment_length):
    """Integra N(t) para UN segmento usando RK4 explícito en malla uniforme."""
    Nt = len(TimeVec)
    N = np.zeros(Nt)
    N[0] = Ninit
    for k in range(Nt - 1):
        II_k, II_k1 = I[k], I[k+1]
        II_mid = 0.5 * (II_k + II_k1)
        PP_k, PP_k1 = Pseg[k], Pseg[k+1]
        PP_mid = 0.5 * (PP_k + PP_k1)
        y = N[k]
        
        def deriv(Ni, cur, pwr):
            RR = A*Ni + B*(Ni**2) + C*(Ni**3)
            Stim = Gamma * DiffGain * (Ni - N0) * pwr * segment_length / (Vol_seg * Ener)
            return cur / (q * Vol_total) - RR - Stim

        k1 = deriv(y, II_k, PP_k)
        k2 = deriv(y + 0.5*dt*k1, II_mid, PP_mid)
        k3 = deriv(y + 0.5*dt*k2, II_mid, PP_mid)
        k4 = deriv(y + dt*k3, II_k1, PP_k1)
        N[k+1] = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return N

def raised_cosine_design(beta, span, spb):
    """Equivalente a rcosdesign(beta, span, spb, 'normal') de MATLAB."""
    n = np.arange(-span * spb / 2, span * spb / 2 + 1)
    h = np.zeros_like(n, dtype=float)
    for i, t in enumerate(n):
        t_norm = t / spb
        if np.abs(t) < 1e-10:
            h[i] = 1.0
        elif np.abs(np.abs(2 * beta * t_norm) - 1.0) < 1e-9:
            h[i] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))
        else:
            h[i] = np.sinc(t_norm) * np.cos(np.pi * beta * t_norm) / (1.0 - (2.0 * beta * t_norm)**2)
    return h / np.sqrt(np.sum(h**2))

# ¡EL FIX! Ahora recibe 'p'
def run_simulation(p):
    # ¡EL FIX! Usa la semilla del objeto p
    np.random.seed(p.semilla) 
    
    # 1. Secuencia de bits y PAM-4 Gray
    bits = np.random.randint(0, 2, p.n_bits)
    UI = p.samples_per_bit
    n_syms = len(bits) // 2
    b2, b1 = bits[0::2], bits[1::2]
    
    sym = np.zeros(n_syms, dtype=int)
    sym[(b2==0) & (b1==1)] = 1
    sym[(b2==1) & (b1==1)] = 2
    sym[(b2==1) & (b1==0)] = 3
    
    sym_mod = 2 * sym - 3
    sym_up = np.zeros(2 * len(sym_mod))
    sym_up[0::2] = sym_mod
    sym2 = lfilter([1.0, 1.0], [1.0], sym_up)
    
    # Generación de pulsos Raised Cosine
    elec_imp = np.zeros(p.n_samples)
    elec_imp[UI // 2::UI] = sym 
    # ¡EL FIX! Usa el beta_rc del objeto p
    h_rc = raised_cosine_design(beta=p.beta_rc, span=8, spb=UI)
    filtered_signal = convolve(elec_imp, h_rc, mode='same')[:p.n_samples]
    
    time_vec = np.arange(p.n_samples) * p.sample_period
    
    # Ajuste de corriente (Bias de 0.45 A asumido)
    # ¡EL FIX! Usa el rango de corriente del objeto p para calcular Imin e Imax
    Imin = 0.45 - (p.rango_corriente / 2.0)
    Imax = 0.45 + (p.rango_corriente / 2.0)
    
    gd_rc = (8 * UI) // 2
    x_ss = filtered_signal[gd_rc:-gd_rc]
    x_lo, x_hi = np.percentile(x_ss, 0.1), np.percentile(x_ss, 99.9)
    m = (Imax - Imin) / (x_hi - x_lo + 1e-12) # Evitar division por cero
    c = Imin - m * x_lo
    I_current = c + m * filtered_signal
    
    # 2. CW Laser de entrada
    e_in = cw_laser(p.n_samples, p.sample_period)
    
    # 3. Solución Segmentada
    num_segments = 10
    segment_length = p.L / num_segments
    Vol_seg = p.width * p.depth * segment_length
    q = 1.602176634e-19
    Ener = 6.62607015e-34 * 299792458 / p.lambda_val
    
    p_seg_in = np.abs(e_in)**2
    e_field = e_in.copy()
    
    for k in range(num_segments):
        N_seg = SOA_N_RK4_uniform(time_vec, I_current, p_seg_in, 0.4e24, p.sample_period, q, Ener, Vol_seg, p.Vol, p.Gamma, p.N0, p.DiffGain, p.A, p.B, p.C, segment_length)
        gan = p.Gamma * p.DiffGain * (N_seg - p.N0)
        e_field *= np.exp((0.5 * (1 - 1j * p.LEF) * gan - 0.5 * p.loss) * segment_length)
        p_seg_in = np.abs(e_field)**2
        
    p_out = p_seg_in
    
    # 4. Alineación y Recorte
    n_skip_sym = 0 
    start0 = n_skip_sym * UI
    corr = correlate(p_out[start0:] - np.mean(p_out[start0:]), I_current[start0:] - np.mean(I_current[start0:]))
    d_soa = np.argmax(corr) - (len(I_current[start0:]) - 1)
    
    if d_soa >= 0:
        idx_I, idx_P = np.arange(0, p.n_samples - d_soa), np.arange(d_soa, p.n_samples)
    else:
        idx_I, idx_P = np.arange(-d_soa, p.n_samples), np.arange(0, p.n_samples + d_soa)
        
    idx_trim = np.arange(start0, len(idx_I))
    I_trim, P_out_trim = I_current[idx_I[idx_trim]], p_out[idx_P[idx_trim]]
    orig_idx_surv = idx_I[idx_trim]

    P_out_trim2 = P_out_trim[0::16] 

    # Barrido de Offset (Métrica Q óptima)
    best_metric, off_best = -np.inf, 0
    for off in range(UI):
        samp_orig = np.arange(n_syms) * UI + off
        mask = np.isin(samp_orig, orig_idx_surv)
        if np.sum(mask) < 20: continue
        
        P_samp_off = P_out_trim[np.searchsorted(orig_idx_surv, samp_orig[mask])] * 1e3
        sym_tx_off = sym[mask]
        mu_off = [np.mean(P_samp_off[sym_tx_off == i]) for i in range(4)]
        sg_off = [np.std(P_samp_off[sym_tx_off == i]) for i in range(4)]
        
        if any(np.isnan(mu_off)) or any(s <= 0 for s in sg_off): continue
        qs = [(mu_off[i+1]-mu_off[i])/(sg_off[i]+sg_off[i+1]) for i in range(3)]
        if min(qs) > best_metric: best_metric, off_best = min(qs), off

    # 5. BER
    samp_final = np.arange(n_syms) * UI + off_best
    mask_f = np.isin(samp_final, orig_idx_surv)
    P_samp = P_out_trim[np.searchsorted(orig_idx_surv, samp_final[mask_f])] * 1e3
    sym_tx = sym[mask_f]
    
    P_levels = [P_samp[sym_tx == i] for i in range(4)]
    mu = np.array([np.mean(lv) for lv in P_levels])
    sg = np.array([np.std(lv) for lv in P_levels])
    
    order_est = np.argsort(mu)
    mu_s, sg_s = mu[order_est], sg[order_est]
    
    Th = [(sg_s[i]*mu_s[i+1] + sg_s[i+1]*mu_s[i])/(sg_s[i]+sg_s[i+1]) for i in range(3)]
    
    sym_hat_sorted = np.digitize(P_samp, Th)
    sym_hat = order_est[sym_hat_sorted]
    
    def sym2bits(s_arr):
        return np.column_stack(((s_arr == 2) | (s_arr == 3), (s_arr == 1) | (s_arr == 2))).ravel()
        
    BER = np.mean(sym2bits(sym_tx) != sym2bits(sym_hat))
    print(f"-> BER Físico: {BER:.3e} | Offset: {off_best}")
    
    return I_trim, P_out_trim, sym2, P_out_trim2

# =====================================================================
# 3. GENERADOR MASIVO (Antes en parquetGenerator.py)
# =====================================================================
def get_parameter_grid():
    bit_rates = [1e9,  8e9, 9e9, 10e9, 15e9]
    betas_rc = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rangos_corr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    semillas = [4, 42]
    num_bits = [2**15]

    keys = ['bit_rate', 'beta_rc', 'rango_corriente', 'semilla', 'num_bits']
    combinations = list(itertools.product(bit_rates, betas_rc, rangos_corr, semillas, num_bits))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def run_massive_generator():
    grid = get_parameter_grid()
    total_combinations = len(grid)
    
    print(f"\n=======================================================")
    print(f"🔥 INICIANDO GENERADOR TODO EN UNO (RAMA SOA5) 🔥")
    print(f"Total de escenarios a simular: {total_combinations}")
    
    out_dir = "HighQualityDataset_SOA5.parquet"
    os.makedirs(out_dir, exist_ok=True)
        
    start_time_total = time.time()
    
    for i, config in enumerate(grid):
        print(f"\n[{i+1}/{total_combinations}] Tasa: {config['bit_rate']/1e9}G | B_rc: {config['beta_rc']} | Rango: {config['rango_corriente']} | Seed: {config['semilla']}")
        
        # 1. Creamos y configuramos el objeto de parámetros
        p = SOAparams()
        p.bit_rate = config['bit_rate']
        p.n_bits = config['num_bits']
        p.beta_rc = config['beta_rc']
        p.rango_corriente = config['rango_corriente']
        p.semilla = config['semilla']
        
        # Recalculamos dependencias de tiempo en base a los nuevos valores
        p.n_samples = int((p.n_bits / 2) * p.samples_per_bit)
        p.fs = p.samples_per_bit * p.bit_rate
        p.sample_period = 1.0 / p.fs

        try:
            # 2. ¡EL FIX! Le pasamos el objeto 'p' a la simulación
            I_trim, P_out_trim, sym2, P_out_trim2 = run_simulation(p)
            
            longitud_minima = min(len(I_trim), len(P_out_trim), len(sym2), len(P_out_trim2))
            
            # 3. Guardamos los resultados
            df = pd.DataFrame({
                'Current_Input_A': I_trim[:longitud_minima],
                'Input_Sym2': sym2[:longitud_minima],
                'Output_P2': P_out_trim2[:longitud_minima] * 1e3,
                
                # Constantes
                'Bit_Rate': config['bit_rate'],
                'Beta_RC': config['beta_rc'],
                'Rango_Corr': config['rango_corriente'],
            })
            
            part_file = f"{out_dir}/part_{i:04d}.parquet"
            df.to_parquet(part_file, engine='pyarrow', index=False)
            
        except Exception as e:
            print(f"❌ Error en la iteración {i}: {e}")
            
    print(f"\n✅ ¡Fierro! Terminado en {(time.time() - start_time_total)/60:.2f} minutos.")

if __name__ == "__main__":
    run_massive_generator()