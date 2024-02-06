# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:22:51 2024

@author: LENOVO
"""

# Importar librerias
import numpy as np
from scipy.optimize import fsolve
import pandas as pd


class PVModel:
    """
    Clase para el modelo de un panel fotovoltaico.
    """

    def __init__(self, num_panels_series, num_panels_parallel):
        self.R_sh = 545.82  # Resistencia en paralelo
        self.k_i = 0.037  # Coeficiente de temperatura
        self.T_n = 298  # Temperatura de referencia
        self.q = 1.60217646e-19  # Carga del electrón
        self.n = 1.0  # Factor de idealidad
        self.K = 1.3806503e-23  # Constante de Boltzmann
        self.E_g0 = 1.1  # Energía de banda prohibida
        self.R_s = 0.39  # Resistencia en serie
        self.num_panels_series = num_panels_series  # Número de paneles en serie
        self.num_panels_parallel = num_panels_parallel  # Número de paneles en paralelo
        # Ajustar los valores de I_sc, V_oc y N_s con el número de paneles en serie y en paralelo
        self.I_sc = 9.35 * num_panels_parallel  # Corriente de cortocircuito
        self.V_oc = 47.4 * num_panels_series  # Voltaje de circuito abierto
        self.N_s = 72 * num_panels_series  # Número de células en serie

    def validate_inputs(self, G, T):
        """
        Validar los valores de irradiancia y temperatura.
        :param G:  Irradiancia (W/m²)
        :param T:  Temperatura (K)
        :return:  None
        """
        if not isinstance(G, (int, float)) or G <= 0:
            raise ValueError("La irradiancia (G) debe ser un número positivo.")
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError("La temperatura (T) debe ser un número positivo.")
        if not isinstance(self.num_panels_series, int) or self.num_panels_series <= 0:
            raise ValueError("El número de paneles en serie debe ser un entero positivo.")
        if not isinstance(self.num_panels_parallel, int) or self.num_panels_parallel <= 0:
            raise ValueError("El número de paneles en paralelo debe ser un entero positivo.")

    def modelo_pv(self, G, T):
        """
        Modelo de un panel fotovoltaico.
        :param G:  Irradiancia (W/m²)
        :param T:  Temperatura (K)
        :return:  DataFrame con los resultados, voltaje, corriente y potencia máximos
        """
        # Validar los valores de irradiancia y temperatura
        self.validate_inputs(G, T)
        # Cálculo de I_rs: corriente de saturación inversa
        I_rs = self.I_sc / (np.exp((self.q * self.V_oc) / (self.n * self.N_s * self.K * T)) - 1)
        # Cálculo de I_o: corriente de saturación inversa
        I_o = I_rs * (T / self.T_n) * np.exp((self.q * self.E_g0 * (1 / self.T_n - 1 / T)) / (self.n * self.K))
        # Cálculo de I_ph: corriente fotogenerada
        I_ph = (self.I_sc + self.k_i * (T - 298)) * (G / 1000)
        # Creación de un vector de voltaje desde 0 hasta V_oc con 1000 puntos
        Vpv = np.linspace(0, self.V_oc, 1000)
        # Inicialización de vectores de corriente y potencia
        Ipv = np.zeros_like(Vpv)
        Ppv = np.zeros_like(Vpv)

        # Función para la ecuación del modelo PV
        def f(I, V):
            return (I_ph - I_o * (np.exp((self.q * (V + I * self.R_s)) / (self.n * self.K * self.N_s * T)) - 1) -
                    (V + I * self.R_s) / self.R_sh - I)
        # Cálculo de la corriente para todo el array de voltaje usando fsolve y vectorización
        Ipv = fsolve(f, self.I_sc * np.ones_like(Vpv), args=(Vpv))
        Ppv = Vpv * Ipv  # Cálculo vectorizado de la potencia

        # Creación de un DataFrame con resultados
        
        resultados = pd.DataFrame({'Corriente (A)': Ipv, 'Voltaje (V)': Vpv, 'Potencia (W)': Ppv})
        # Encontrar el punto de máxima potencia
        max_power_idx = resultados['Potencia (W)'].idxmax()
        Vmpp = resultados.loc[max_power_idx, 'Voltaje (V)']
        Impp = resultados.loc[max_power_idx, 'Corriente (A)']
        P_max = resultados.loc[max_power_idx, 'Potencia (W)']
        return resultados, Vmpp, Impp, P_max

def main():
    # Crear un objeto de la clase PVModel
    pv = PVModel(4,3)
    # Calcular el modelo PV
    resultados, Vmpp, Impp, P_max = pv.modelo_pv(G=1000, T=273+25)
    print(resultados.head())
    print(f"Vmp = {Vmpp:.2f} V, Imp = {Impp:.2f} A, Pmax = {P_max:.2f} W")

if __name__ == "__main__":
    main()

#### Paso 1 Generar Datos y calcular valores

import random
import sqlite3


def generar_y_almacenar_datos():
    pv = PVModel(4, 3)  # Asumiendo 4 paneles en serie y 3 en paralelo
    conn = sqlite3.connect('pv_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS pv_data
                      (G REAL, T REAL, Vmp REAL, Imp REAL, Pmax REAL)''')

    for _ in range(10):  # Generar 10 conjuntos de datos
        G = random.uniform(800, 1200)  # Irradiancia aleatoria entre 800 y 1200
        T = random.uniform(15, 35)  # Temperatura aleatoria entre 15 y 35 grados Celsius
        _, Vmpp, Impp, P_max = pv.modelo_pv(G, T + 273.15)  # Convertir T a Kelvin
        cursor.execute('INSERT INTO pv_data (G, T, Vmp, Imp, Pmax) VALUES (?, ?, ?, ?, ?)',
                       (G, T, Vmpp, Impp, P_max))

    conn.commit()
    conn.close()

# Ejecutar la función para generar y almacenar datos
generar_y_almacenar_datos()
print("generar_y_almacenar_datos()")

## Paso 2. Configuración de la Aplicación Flask

from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Esta ruta mostrará un formulario para ingresar nuevos datos
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Obtener datos del formulario
    G = float(request.form['irradiance'])
    T = float(request.form['temperature'])
    
    # Calcular los resultados con PVModel
    pv = PVModel(4, 3)
    resultados, Vmpp, Impp, P_max = pv.modelo_pv(G, T + 273.15)
    
    # Generar la gráfica y pasar los resultados a la plantilla de resultados
    # La función para generar la gráfica se explicará en el siguiente paso
    plot_url = generate_plot(resultados)
    return render_template('results.html', plot_url=plot_url, Vmpp=Vmpp, Impp=Impp, P_max=P_max)

##3. Creación de Gráficas y Visualización en la Página Web

'''
def generate_plot(resultados):
    plt.figure()
    plt.plot(resultados['Voltaje (V)'], resultados['Corriente (A)'], label='Imp vs Vmp')
    plt.plot(resultados['Voltaje (V)'], resultados['Potencia (W)'], label='Pmax vs Vmp')
    plt.xlabel('Voltaje (V)')
    plt.ylabel('Corriente (A) / Potencia (W)')
    plt.title('Curvas I-V y P-V del Panel Fotovoltaico')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
    pass
'''

def generate_plot(resultados):     
    plt.figure()     
    # Normalización de Corriente (A) y Potencia (W)    
    max_corriente = resultados['Corriente (A)'].max()     
    max_potencia = resultados['Potencia (W)'].max()     
    corriente_normalizada = resultados['Corriente (A)'] / max_corriente     
    potencia_normalizada = resultados['Potencia (W)'] / max_potencia     
    # Graficar Corriente y Potencia normalizadas    
    plt.plot(resultados['Voltaje (V)'], corriente_normalizada, label='Imp vs Vmp (normalizado)')     
    plt.plot(resultados['Voltaje (V)'], potencia_normalizada, label='Pmax vs Vmp (normalizado)')     
    plt.xlabel('Voltaje (V)')     
    plt.ylabel('Valor Normalizado')     
    plt.title('Curvas I-V y P-V Normalizadas del Panel Fotovoltaico')     
    plt.legend()
    plt.grid()     
    img = io.BytesIO()     
    plt.savefig(img, format='png', bbox_inches='tight')     
    img.seek(0)     
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')     
    return plot_url
 

if __name__ == '__main__':
    app.run(debug=True)