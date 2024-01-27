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
from io import BytesIO
import time
# Librerías de terceros
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras
# Librerías propias del proyecto
from streamlit_func import imagen_con_enlace

# Funciones auxiliares
@st.cache_resource()
def load_model() -> keras.Model:
    return keras.models.load_model("models/convnet_mnist_104k.keras")

def process_image():
    pass

def main() -> None:
    """Entry point de la app"""

    # Configuración de la app
    st.set_page_config(
        page_title=f"Reconocimiento de dígitos",
        page_icon="👁️", 
        layout="wide",
        initial_sidebar_state="auto",
    )

    with st.sidebar:
        # Imagen de la app
        #st.image('img/logo_app.png')
        imagen_con_enlace('https://i.imgur.com/4f38x2v.png', 'https://kopuru.com/', centrar=True)
        st.caption('© 2024 STM')

    st.title('Reconocimiento de dígitos')
    st.subheader('Una App para Kopuru')
    st.write('''Con esta app serás capaz de evaluar un modelo convolucional entrenado 
                con el dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
                Dibuja un dígito en color blanco sobre fondo negro en una imagen de 28x28 píxeles
                y súbela. Pulsa sobre el botón **predecir** y comprueba si el modelo ha sido
                capaz de averiguar el dígito que habías dibujado.''')
    
    # Inicializamos variables de sesión para llevar un registro de las predicciones
    if st.session_state.get('predicciones') is None:
        st.session_state['predicciones'] = []

    tab1, tab2, tab3, tab4 = st.tabs(['Cargar imagen', 'Ver dígito', 'Predecir', 'Ver historial'])

    with tab1:
        st.write('''Carga tu imagen con el dígito dibujado. 
            Recuerda que debe ser una imagen de 28x28 píxeles. El dígito debe estar
            dibujado en blanco sobre color negro.''')
        
        imagen_bruta = st.file_uploader('Sube tu dígito', type=["png","tif","jpg","bmp","jpeg"])

        # TODO validaciones

        if imagen_bruta is not None: #TODO and validaciones
            img_array = np.array(Image.open(BytesIO(imagen_bruta.read())))
            X = (img_array.reshape(1,28,28,1).astype(np.float32) - 127.5 ) / (127.5)
            

    with tab2:
        # TODO plotear con matplotlib imshow el dígito
        pass

    with tab3:
        # TODO Comprobamos que haya una imagen válida cargada
        predecir = st.button("Predecir")
        if predecir:
            model = load_model()
            with st.spinner(text="Prediciendo dígito..."):
                pred = model.predict(X)
                probs = round(np.max(model.predict_proba(X)) * 100, 2)
                time.sleep(1)
            st.metric("Predicción del modelo", value=pred, delta=f"{'-' if probs < 70 else ''}{probs:.2%}")
        
            st.subheader("¿ Ha acertado el modelo ?")
            st.number_input("Marca el dígito que habías dibujado", min_value=0, max_value=9)

            guardar_pred = st.button("Guardar predicción", help='Añade los valores al historial')
            if guardar_pred:
                # TODO No guardar si nombre de archivo existe en historial
                # Guardamos los valores de la predicción en sesión en un dict
                prediccion = {
                    'nombre_archivo': imagen_bruta.name,
                    'prediccion_modelo': "",
                }
                st.session_state['predicciones'].append()

    with tab4:
        # TODO Mostrar acumulado de las predicciones: +1 acierto -1 fallo
        # TODO Mostrar porcentaje de aciertos
        pass

        

    st.session_state

if __name__ == '__main__':
    main()
    