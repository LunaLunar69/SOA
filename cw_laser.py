import numpy as np

def cw_laser(n_samples, sample_period):
    # Parámetros actualizados
    avg_pow_mw = 10         # Potencia reducida a 10 mW
    df = 50e3               # Half laser linewidth en Hz
    
    avg_pow_w = avg_pow_mw / 1000.0
    
    # Generación de ruido blanco gaussiano para la fase
    w = np.random.randn(n_samples) * (2 * np.pi * df)
    
    # Integración para obtener fase (cumsum)
    phi = np.cumsum(w) * sample_period
    
    # Campo eléctrico (Complex envelope)
    elec_field = np.sqrt(avg_pow_w) * (np.cos(phi) + 1j * np.sin(phi))
    
    return elec_field