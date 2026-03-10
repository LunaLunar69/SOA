import numpy as np

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

        # Signal Parameters (Actualizados según MATLAB reciente)
        self.Numsymrrc = 8
        self.bit_rate = 5.35e9          # Tasa de bits reducida
        self.n_bits = 2**14             # Mayor estadística
        self.samples_per_bit = 32
        
        # Cálculos derivados de la señal
        self.n_samples = int((self.n_bits / 2) * self.samples_per_bit)
        self.fs = self.samples_per_bit * self.bit_rate
        self.sample_period = 1.0 / self.fs

        # Nuevos parámetros dinámicos para la generacion de dataset
        self.beta_rc = 0.5
        self.rango_corriente = 0.4
        self.semilla = 42