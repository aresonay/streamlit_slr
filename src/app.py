import streamlit as st 
import pandas as pd 
import joblib 
import matplotlib.pyplot as plt 
import numpy as np 


slr_model = joblib.load("slregression_model.pkl")

st.title("Regresión Lineal Simple")

st.write("""
    Modelo de regresión lineal simple aplicado a un conjunto de 
    datos con los datos del sueldo en función de los años de 
    experiencia
""")


st.header("Muestra con x años de experiencia aleatorios")


muestra = np.random.uniform(1,11, size=40)

muestra = np.array(list(map(lambda n: round(n, 1), muestra)))

st.code(muestra)

st.write("""
    En esta gráfica puedes ver como los salarios se ajustan 
    perfectamente a la línea de regresión trazada por los datos 
    del modelo. Cada dato que recibe se ajustará al punto que le 
    corresponda a lo largo de esta recta. 
""")

st.write("""
    Prueba tú mismo a generar una muestra aleatoria para observar
    los resultados. 
""")

tamano_muestra = st.slider(
    'Slider1', 
    1, 100,
    label_visibility="collapsed",
    key="sample_size")

random_sample = st.button(label="Generar Muestra", key="random_button")

if random_sample:
    muestra = np.random.uniform(1, 11, size=tamano_muestra)
    muestra = np.array(list(map(lambda n: round(n,1), muestra)))

prediccion_muestra = slr_model.predict(muestra.reshape(-1, 1))

valores_regresion = np.array([[ 2.9],
       [ 5.1],
       [ 3.2],
       [ 4.5],
       [ 8.2],
       [ 6.8],
       [ 1.3],
       [10.5],
       [ 3. ],
       [ 2.2],
       [ 5.9],
       [ 6. ],
       [ 3.7],
       [ 3.2],
       [ 9. ],
       [ 2. ],
       [ 1.1],
       [ 7.1],
       [ 4.9],
       [ 4. ]])


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.scatter(muestra, prediccion_muestra.reshape(-1, 1), color="red")
plt.plot(valores_regresion, slr_model.predict(valores_regresion), color="blue")
plt.grid(True)
plt.title("Salario vs Años de experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario en $")
st.pyplot()
with st.sidebar: 
    st.header("Relación de datos Años Experiencia vs Salario")
    tabla_datos = {'Años de experiencia': muestra, 'Salario $': prediccion_muestra}
    tabla = pd.DataFrame(data=tabla_datos)
    tabla_ordenada = tabla.sort_values(by=['Años de experiencia'])
    st.table(tabla_ordenada)

sueldo = st.number_input("Introduce años de experiencia")

btn = st.button(label='Estimar sueldo $')

if btn:
    sueldo = np.array([[sueldo]])
    prediction = slr_model.predict(sueldo)
    st.header("{:.2f}".format(prediction[0]))
