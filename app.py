# Copyright 2024 Sergio Tejedor Moreno

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Librerías internas

# Librerías de terceros
import streamlit as st

# Configuración de la app
st.set_page_config(
    page_title=f"Reconocimiento de dígitos",
    page_icon="👁️", 
    layout="centered",
    initial_sidebar_state="auto",
)

with st.sidebar:
    st.write('© 2024 STM')

st.title('Reconocimiento de dígitos')
st.subheader('Una App para Kopuru')
st.write('''Con esta app serás capaz de evaluar un modelo convolucional entrenado 
            con el dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
            Dibuja un dígito en color blanco sobre fondo negro en una imagen de 28x28 píxeles
            y súbela. Pulsa sobre el botón **predecir** y comprueba si el modelo ha sido
            capaz de averiguar el dígito que habías dibujado.''')

tab1, tab2, tab3, tab4 = st.tabs(['Cargar imagen', 'Ver dígito', 'Ver prediccion', 'Ver historial'])

with tab1:
    st.write('''Carga tu imagen con el dígito dibujado. 
        Recuerda que debe ser una imagen de 28x28 píxeles. El dígito debe estar
        dibujado en blanco sobre color negro.''')
    
    imagen = st.file_uploader('Sube tu dígito', type=["png","tif","jpg","bmp","jpeg"])

    # TODO validaciones

    if imagen is not None: #TODO and validaciones
        predecir = st.button("Predecir")

        if predecir:
            # TODO barra de progreso ?
            # TODO lanzar predicción
            pass

with tab2:
    # TODO plotear con matplotlib imshow el dígito
    pass

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicción del modelo", value=2)
    with col2:
        st.metric("Confianza de la predicción", value=88)
    
    st.write("¿ Ha acertado el modelo ?")
    st.number_input("Marca el dígito que habías dibujado", min_value=0, max_value=9)

    guardar_pred = st.button("Guardar valores", help='Añade los valores al historial')
    if guardar_pred:
        # TODO No guardar si nombre de archivo existe en historial
        pass

with tab4:
    # TODO Mostrar acumulado de las predicciones: +1 acierto -1 fallo
    # TODO Mostrar porcentaje de aciertos
    pass

    

st.session_state
    