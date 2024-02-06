<div align="center">
	<h1 style="color:Tomato;"><strong>Taller Final - Modelo Fotovoltaico</strong></h1>
	<strong>Realizado por:</strong> Diogenes Barreto Alvarez<br>
  Carlos Mauricio Moreno Rojas<br>
  Miller Alexander Quiroga Campos<br>
  John Mario Gutierrez<br>
</div>
<br><hr>
<div align="center">
En el contexto del taller final del modelo fotovoltaico, la estructura del entorno de desarrollo se presenta de la siguiente manera:
</div>

<br><hr>
PV_Model<br>
├─ templates<br>
│  ├─ index.html<br>
│  └─ results.html<br>
│<br>
├─ PV_Model.py<br>
└─ pv_data.db<br>
<br><hr>

- Dentro del archivo principal en Python (PV_Model.py), se lleva a cabo la importación de las librerías esenciales para la implementación:

<br><hr>
import numpy as np<br>
from scipy.optimize import fsolve<br>
import pandas as pd<br>
import random<br>
import sqlite3<br>
from flask import Flask, request, render_template<br>
import matplotlib.pyplot as plt<br>
import io<br>
import base64<br>
<br><hr>


- A continuación, se define la clase PVModel que encapsula la construcción del modelo fotovoltaico, basado en los principios establecidos en la tesis titulada "Control por modos deslizantes aplicado a un inversor de fuente de corriente monofásico conectado a la red".

- Una vez construido el modelo, se generan datos de manera aleatoria y se calculan los valores correspondientes para cada variable del modelo.

- La configuración de la aplicación Flask se lleva a cabo, definiendo dos rutas principales: una para la visualización de datos existentes y otra que permite al usuario ingresar nuevos datos como Irradiancia (G) y Temperatura (T), proporcionando los resultados correspondientes (index.html - results.html).

- Se realiza la normalización de la Corriente (Impp) y la Potencia (P_max), seguido por la generación de gráficos y su visualización en la página web. Este enfoque técnico e ingenieril garantiza la representación precisa y eficiente de los resultados del modelo fotovoltaico en un formato accesible para el usuario final.
<br><hr>
  

