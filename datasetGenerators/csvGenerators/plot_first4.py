import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_single_eye_with_hist(ax_eye, ax_hist, df_escenario, UI, title, unit_scale):
    # Extraemos la señal de salida (Ya no leemos Time_s)
    signal = df_escenario['Power_Output_mW'].values
    
    # ⏱️ LA MAGIA: Calculamos el tiempo real al vuelo usando el Bit Rate de esta señal
    bit_rate = df_escenario['Bit_Rate'].iloc[0]
    dt = 1.0 / (bit_rate * UI)
    
    # Parámetros del ojo
    span_ui = 4
    samples_span = int(span_ui * UI)
    offset_samples = UI // 2
    
    # Recorte inicial
    sig_s = signal[offset_samples:]
    n_traces = len(sig_s) // samples_span
    
    if n_traces < 1:
        print(f"Advertencia: No hay suficientes datos para {title}")
        return

    # Acomodamos la señal para el diagrama de ojo
    sig_to_reshape = sig_s[:n_traces * samples_span]
    eye_m = (sig_to_reshape * unit_scale).reshape((n_traces, samples_span))
    
    # Creamos el vector de tiempo en picosegundos para el eje X
    t_span = np.linspace(-span_ui/2 * UI * dt, span_ui/2 * UI * dt, samples_span) * 1e12 # Convertido a ps
    
    # Graficar trazas en el ojo
    ax_eye.plot(t_span, eye_m.T, color='#FFFF00', alpha=0.3, linewidth=0.5)
    
    # Estética del Ojo
    ax_eye.set_facecolor('black')
    ax_eye.set_title(title, color='white', fontsize=10)
    ax_eye.set_ylabel('$P_{out}$ (mW)', color='white')
    ax_eye.set_xlabel('Tiempo (ps)', color='white')
    ax_eye.tick_params(colors='white')
    ax_eye.grid(True, color='gray', alpha=0.3)
    
    # Histograma a la derecha
    ax_hist.hist(eye_m[:, samples_span // 2], bins=100, color='#FFFF00', alpha=0.6, orientation='horizontal')
    ax_hist.set_facecolor('black')
    ax_hist.axis('off')

def main():
    print("⏳ Leyendo el monstruo... (solo extrayendo el inicio para no matar la RAM)")
    
    # Leemos solo 2.2 millones de filas (Suficiente para abarcar los primeros 4 escenarios completos)
    archivo = "datasets_results/single_dataset/HighQualityDataset.csv"
    df = pd.read_csv(archivo, nrows=2200000)
    
    # Identificar cuándo cambian los parámetros para agrupar por "Escenario"
    cols_config = ['Bit_Rate', 'Beta_RC', 'Rango_Corr']
    df['Escenario'] = (df[cols_config] != df[cols_config].shift()).any(axis=1).cumsum()
    
    # Nos quedamos estrictamente con los primeros 4
    df_primeros_4 = df[df['Escenario'] <= 4]
    
    escenarios_unicos = df_primeros_4['Escenario'].unique()
    UI = 32 # Samples per bit
    
    # Preparamos la Mega Figura
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    gs = gridspec.GridSpec(4, 2, width_ratios=[4, 1], wspace=0.05, hspace=0.5)
    
    print(f"✅ Se encontraron {len(escenarios_unicos)} escenarios. ¡Graficando!")
    
    for i, esc in enumerate(escenarios_unicos):
        df_esc = df_primeros_4[df_primeros_4['Escenario'] == esc]
        
        # Sacamos los parámetros para ponerlos en el título
        br = df_esc['Bit_Rate'].iloc[0] / 1e9  # En Gbps
        beta = df_esc['Beta_RC'].iloc[0]
        rc = df_esc['Rango_Corr'].iloc[0]
        
        titulo = f"Escenario {i+1} | BR: {br} Gbps | Beta: {beta} | Rango I: {rc} A"
        
        # Asignamos los "cuadritos" de la gráfica
        ax_eye = plt.subplot(gs[i, 0])
        ax_hist = plt.subplot(gs[i, 1], sharey=ax_eye)
        
        # La señal del dataset ya viene en mW, así que la escala es 1.0
        plot_single_eye_with_hist(ax_eye, ax_hist, df_esc, UI, titulo, unit_scale=1.0)
        
        # Quitamos los números del eje Y del histograma para que se vea limpio
        ax_hist.tick_params(labelleft=False, left=False)

    # Ajustamos y mostramos
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97]) 
    plt.show()

if __name__ == "__main__":
    main()