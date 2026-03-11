import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate, convolve
from numba import njit
from params import SOAparams
from cw_laser import cw_laser

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

def run_simulation(p):
    t_start = time.time()
    
    # 1. Inyección de la Semilla Dinámica
    semilla = getattr(p, 'semilla', 4)
    np.random.seed(semilla) 
    
    # Secuencia de bits y PAM-4 Gray
    bits = np.random.randint(0, 2, p.n_bits)
    UI = p.samples_per_bit
    n_syms = len(bits) // 2
    b2, b1 = bits[0::2], bits[1::2]
    
    sym = np.zeros(n_syms, dtype=int)
    sym[(b2==0) & (b1==1)] = 1
    sym[(b2==1) & (b1==1)] = 2
    sym[(b2==1) & (b1==0)] = 3
    
    # --- NUEVO: Generación de pulsos Raised Cosine ---
    elec_imp = np.zeros(p.n_samples)
    elec_imp[UI // 2::UI] = sym 
    
    # 2. Inyección del Filtro Beta Dinámico
    beta_rc = getattr(p, 'beta_rc', 0.3)
    h_rc = raised_cosine_design(beta=beta_rc, span=8, spb=UI)
    filtered_signal = convolve(elec_imp, h_rc, mode='same')[:p.n_samples]
    
    time_vec = np.arange(p.n_samples) * p.sample_period
    
    # 3. Inyección del Rango de Corriente Dinámico
    rango = getattr(p, 'rango_corriente', 0.4)
    Ibias = 0.45 
    Imin = Ibias - (rango / 2.0)
    Imax = Ibias + (rango / 2.0)
    
    gd_rc = (8 * UI) // 2
    x_ss = filtered_signal[gd_rc:-gd_rc]
    x_lo, x_hi = np.percentile(x_ss, 0.1), np.percentile(x_ss, 99.9)
    m = (Imax - Imin) / (x_hi - x_lo)
    c = Imin - m * x_lo
    I_current = c + m * filtered_signal
    
    # CW Laser de entrada
    e_in = cw_laser(p.n_samples, p.sample_period)
    
    # Solución Segmentada
    num_segments = 10
    segment_length = p.L / num_segments
    Vol_seg = p.width * p.depth * segment_length
    q = 1.602176634e-19
    Ener = 6.62607015e-34 * 299792458 / p.lambda_val
    
    p_seg_in = np.abs(e_in)**2
    e_field = e_in.copy()
    
    print("Simulando propagación en SOA...")
    for k in range(num_segments):
        N_seg = SOA_N_RK4_uniform(time_vec, I_current, p_seg_in, 0.4e24, p.sample_period, q, Ener, Vol_seg, p.Vol, p.Gamma, p.N0, p.DiffGain, p.A, p.B, p.C, segment_length)
        gan = p.Gamma * p.DiffGain * (N_seg - p.N0)
        e_field *= np.exp((0.5 * (1 - 1j * p.LEF) * gan - 0.5 * p.loss) * segment_length)
        p_seg_in = np.abs(e_field)**2
        
    p_out = p_seg_in
    
    # Alineación y Recorte
    n_skip_sym = 2
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

    # --- NUEVO: Barrido de Offset (Métrica Q óptima) ---
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

    # 5. BER, SER y Umbrales con off_best
    samp_final = np.arange(n_syms) * UI + off_best
    mask_f = np.isin(samp_final, orig_idx_surv)
    P_samp = P_out_trim[np.searchsorted(orig_idx_surv, samp_final[mask_f])] * 1e3
    sym_tx = sym[mask_f]
    
    P_levels = [P_samp[sym_tx == i] for i in range(4)]
    mu = np.array([np.mean(p) for p in P_levels])
    sg = np.array([np.std(p) for p in P_levels])
    
    order_est = np.argsort(mu)
    mu_s, sg_s = mu[order_est], sg[order_est]
    
    Th = [(sg_s[i]*mu_s[i+1] + sg_s[i+1]*mu_s[i])/(sg_s[i]+sg_s[i+1]) for i in range(3)]
    
    sym_hat_sorted = np.digitize(P_samp, Th)
    sym_hat = order_est[sym_hat_sorted]
    SER = np.mean(sym_hat != sym_tx)
    
    def sym2bits(s_arr):
        return np.column_stack(((s_arr == 2) | (s_arr == 3), (s_arr == 1) | (s_arr == 2))).ravel()
        
    BER = np.mean(sym2bits(sym_tx) != sym2bits(sym_hat))
    print(f"BER = {BER:.3e} | Offset Óptimo: {off_best}")
    
    return I_trim, P_out_trim, P_levels, Th, off_best

def plot_results(I_trim, P_out_trim, P_levels, thresholds, sample_period):
    t_trim = np.arange(len(I_trim)) * sample_period
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1); plt.plot(t_trim, I_trim); plt.ylabel('I [A]'); plt.grid(True)
    plt.subplot(2,1,2); plt.plot(t_trim, P_out_trim * 1e3); plt.ylabel('$P_{out}$ [mW]'); plt.grid(True)
    
    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, P_vals in enumerate(P_levels):
        plt.hist(P_vals, bins=100, alpha=0.7, color=colors[i], label=f'Nivel {i}')
    for i, val in enumerate(thresholds):
        plt.axvline(val, color=['r','c','b'][i], linestyle='--', label=f'Th{i+1}')
    plt.title('Histogramas en Instante Óptimo de Decisión'); plt.legend(); plt.grid(True)

def plot_eye_diagrams(I_trim, P_out_trim, UI, off_best):
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], wspace=0.05, hspace=0.3)
    
    # Marcamos la línea de decisión en el ojo basada en el off_best
    x_dec = (off_best - (UI // 2)) / UI
    
    for i, (sig, lbl, unit) in enumerate([(I_trim, 'I [A]', 1.0), (P_out_trim, '$P_{out}$ [mW]', 1e3)]):
        ax_eye = plt.subplot(gs[i, 0]); ax_hist = plt.subplot(gs[i, 1], sharey=ax_eye)
        plot_single_eye_with_hist(ax_eye, ax_hist, sig, UI, f'Eye Diagram - {lbl}', lbl, unit, UI//2)
        ax_eye.axvline(x_dec, color='r', linestyle='--', linewidth=1.5) # Línea de decisión

def plot_single_eye_with_hist(ax_eye, ax_hist, signal, UI, title, ylabel, unit_scale, offset_samples):
    span_ui = 4; samples_span = span_ui * UI
    sig_s = signal[offset_samples:]
    n_traces = len(sig_s) // samples_span
    eye_m = (sig_s[:n_traces * samples_span] * unit_scale).reshape((n_traces, samples_span))
    t_span = np.linspace(-span_ui/2, span_ui/2, samples_span)
    ax_eye.plot(t_eye := t_span, eye_m.T, color='#FFFF00', alpha=0.3, linewidth=0.5)
    ax_eye.set_facecolor('black'); ax_eye.set_title(title, color='white'); ax_eye.grid(True, color='gray', alpha=0.5)
    ax_hist.hist(eye_m[:, samples_span // 2], bins=100, color='#FFFF00', alpha=0.6, orientation='horizontal')
    ax_hist.set_facecolor('black'); ax_hist.axis('off')

if __name__ == "__main__":
    p = SOAparams()
    # Le pasamos "p" a la simulación
    I_t, P_t, P_l, Th, off = run_simulation(p)
    
    plot_results(I_t, P_t, P_l, Th, p.sample_period)
    
    # Se le pasa la lista de tuplas con las señales como lo pide tu función
    signals_para_ojo = [
        (I_t, 'Driving Current (I)', 1.0),
        (P_t, 'Output Power (P_out)', 1e3)
    ]
    plot_eye_diagrams(signals_para_ojo, p.samples_per_bit, off)
    plt.show()