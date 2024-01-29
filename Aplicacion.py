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
from datetime import datetime
from io import BytesIO
from pathlib import Path
import pytz
import time
from typing import Tuple
# Librerías de terceros
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow import keras
# Librerías propias del proyecto
from streamlit_func import show_sidebar

# Constantes
MODEL_PATH = Path('models')
DAY_HOUR_FORMAT = """%d/%m/%y\n%H:%M"""
DAY_FORMAT = "%d/%m/%y"

# Funciones auxiliares
@st.cache_resource()
def load_model() -> keras.Model:
    """Devuelve el modelo con los weights cargados

    Parameters
    ----------
    model : keras.Model
        Modelo virgen
    weights_file : str
        Archivo *.h5 donde están los weights

    Returns
    -------
    keras.Model
        _description_
    """
    # Construimos el modelo
    model = build_model()
    weights = MODEL_PATH / 'convnet_mnist_104k_weights.h5'
    model.load_weights(weights, by_name=True)
    # Cargamos los weights y devolvemos
    return model

def build_model() -> keras.Model:
    """Reconstruye el modelo y lo devuelve para posteriormente ser cargado
    con los weights del modelo entrenado. Lo tenemos que hacer asi
    debido a incompatibilidades al guardado el modelo original:

    https://github.com/keras-team/keras-core/issues/855

    Returns
    -------
    keras.Model
        Devuelve el modelo con los weights sin entrenar
    """
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def process_image(img_array:np.ndarray) -> np.ndarray:
    """Preprocesa la imagen cargada para adecuarla a los requerimientos
    del modelo

    Parameters
    ----------
    img_array : np.ndarray
        Imagen en formato array sin procesar

    Returns
    -------
    np.ndarray
        Imagen reescalada con el formato adecuado para alimentar el modelo
    """
    img_process = img_array.reshape(1,28,28,1).astype(np.float32) / 255
    return img_process

@st.cache_data()
def predict(img_processed:np.ndarray) -> Tuple[int, float]:
    """Lanza el modelo sobre la imagen procesada
    y devuelve una tupla con la predicción del modelo
    y la confianza

    Parameters
    ----------
    img_processed : np.ndarray
        Imagen en formato array de numpy con el formato
        adecuado

    Returns
    -------
    Tuple[int, float]
        Dígito predicho y confianza
    """
    # Cargamos el modelo
    model = load_model()    
    # Sacamos array de probabilidades
    probs:np.ndarray = model.predict(img_processed)
    # Sacamos la predicción del dígito
    pred = np.argmax(probs)
    # Sacamos la confianza de dicha predicción
    conf = probs.max()
    return pred, conf

def reset_predictions() -> None:
    """Resetea algunas variables de sesión cuando se cambia de imagen cargada
    o se elimina la imagen cargada"""
    if 'ultima_prediccion' in st.session_state:
        del st.session_state['ultima_prediccion']

def get_timestamp(formato:str) -> str:
    """Recibe un formato de timestamp en string y devuelve
    en ese formato el momento del día actual

    Parameters
    ----------
    formato : str
        _description_

    Returns
    -------
    str
        _description_
    """
    return datetime.strftime(datetime.now(tz=pytz.timezone('Europe/Madrid')),format=formato)

# Validaciones
def pred_already_saved(filename:str) -> bool:
    """Comprueba si el nombre de archivo está guardado ya en el historial.
    Devuelve True si ya ha sido guardado y False en caso contrario

    Parameters
    ----------
    filename : str
        _description_

    Returns
    -------
    bool
        _description_
    """
    return any(filename in pred.values() for pred in st.session_state['historial'])

def is_valid_image(img_array:np.ndarray) -> Tuple[bool, str]:
    """Realiza validaciones al array de la imagen.
    Devuelve una tupla con un bool y un mensaje de error

    Parameters
    ----------
    img_array : np.ndarray
        La imagen en formato array de numpy

    Returns
    -------
    Tuple[bool, str]
        True, "" si la imagen es válida
        False, "mensaje de error" si la imagen no es válida
    """
    # Si hay más de un 95% de negro o menos de un 10% consideramos que el número no puede ser válido
    if (np.count_nonzero(img_array == 0) > 0.95 * img_array.size) or (np.count_nonzero(img_array == 0) < 0.10 * img_array.size):
        return False, "La imagen no es válida. Carga una imagen con un dígito en blanco sobre fondo negro."
    # Comprobamos la shape de la imagen, que debe ser 28 x 28 píxeles
    if img_array.shape != (28, 28):
        return False, f"La dimensión de la imagen debe ser (28, 28). La imagen cargada es {img_array.shape}."
    return True, ""


def main() -> None:
    """Entry point de la app"""

    # Configuración de la app
    st.set_page_config(
        page_title=f"Reconocimiento de dígitos",
        page_icon="👁️", 
        layout="centered",
        initial_sidebar_state="auto",
    )
    # Mostramos la Sidebar que hemos configurado en streamlit_func
    show_sidebar()

    st.title('Reconocimiento de dígitos')
    st.subheader('Una App para Kopuru')
    st.write('''Con esta app serás capaz de evaluar un modelo convolucional entrenado 
                con el dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).<br>
                Sube una imagen monocanal (escala de grises) de dimensiones 28x28 píxeles con
                un dígito dibujado en color blanco sobre fondo negro.<br>
                Pulsa sobre el botón **predecir** y comprueba si el modelo ha sido
                capaz de averiguar el dígito que habías dibujado.''', unsafe_allow_html=True)
    
    # Inicializamos variables de sesión para llevar un registro de las predicciones
    if st.session_state.get('historial') is None:
        st.session_state['historial'] = []
    # Flag para saber si tenemos una imagen cargada y validada
    st.session_state['imagen_cargada_y_validada'] = False

    tab_cargar_imagen, tab_ver_digito, tab_predecir, \
        tab_evaluar, tab_historial = st.tabs(['Cargar imagen', 'Ver dígito', 
                                        'Predecir', 'Evaluar', 'Ver historial'])

    with tab_cargar_imagen:
        st.write('''Carga tu imagen con el dígito dibujado. 
            Recuerda que debe ser una imagen de 28x28 píxeles.<br>El dígito debe estar
            dibujado en blanco sobre color negro.''', unsafe_allow_html=True)
        
        imagen_bruta = st.file_uploader('Sube tu dígito', type=["png","tif","jpg","bmp","jpeg"], on_change=reset_predictions)

        if imagen_bruta is not None:
            # Utilizamos el wrapper BytesIO para cargar bytes en la calse Image de PIL
            # Transformamos la imagen en array de numpy
            img_array = np.array(Image.open(BytesIO(imagen_bruta.read())))
            # Realizamos validaciones sobre la imagen
            valid_img, error_msg = is_valid_image(img_array)
            # Si la imagen no es válida mostramos mensaje de error y paramos la ejecución de la app
            # Forzamos al usuario a cargar una nueva imagen válida
            if not valid_img:
                st.error(error_msg)
                st.stop() # Lo que viene después del stop no se ejecutará.

            st.session_state['imagen_cargada_y_validada'] = imagen_bruta.name    
            
    with tab_ver_digito:
        # Verificamos que tengamos una imagen cargada y validada en sesión
        if st.session_state.get('imagen_cargada_y_validada'):
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(img_array, cmap="gray")
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Carga una imagen para visualizar.")

    with tab_predecir:
        # Comprobamos si se ha lanzado una predicción y por tanto está almacenada en sesión.
        # Si tenemos una predicción la mostramos
        if (last_pred:=st.session_state.get('ultima_prediccion')) is not None:
            pred = last_pred['pred']
            conf = last_pred['conf']
            st.metric("Predicción del modelo", value=pred, delta=f"{'-' if conf < 0.7 else ''}{conf:.2%}")
        else:
            # Si no se ha lanzado una predicción, mostramos mecanismo para lanzarla
            # Verificamos que tengamos una imagen cargada y validada en sesión
            if nombre_imagen:=st.session_state.get('imagen_cargada_y_validada'):
                predecir = st.button(f"Predecir {nombre_imagen}")
                if predecir:
                    # Procesamos la imagen
                    img_processed = process_image(img_array)
                    # Lanzamos las predicciones
                    with st.spinner(text="Prediciendo dígito..."):
                        pred, conf = predict(img_processed)
                        time.sleep(1)
                    st.metric("Predicción del modelo", value=pred, delta=f"{'-' if conf < 0.7 else ''}{conf:.2%}")
                    # Guardamos en sesión
                    st.session_state['ultima_prediccion'] = {
                        'pred': int(pred),
                        'conf': conf,
                        'archivo': nombre_imagen,
                    }
                    
            else:
                st.info("Carga una imagen para predecir.")

    with tab_evaluar:
        # Verificamos si hay una predicción lanzada y guardada en sesión
        if (last_pred:=st.session_state.get('ultima_prediccion')) is not None:
            # Posibilidad de contrastar con la realidad para almacenar porcentaje de aciertos
            st.subheader("¿ Ha acertado el modelo ?")
            digit = st.number_input("Marca el dígito que habías dibujado", min_value=0, max_value=9)
            guardar_pred = st.button("Guardar evaluación", help='Añade la evaluación al historial')
            if guardar_pred:
                # Comprobamos si ya está ese archivo guardado
                if not pred_already_saved(imagen_bruta.name):
                    # Añadimos a ultima_prediccion la evaluación del usuario
                    last_pred['real'] = digit
                    # Añadimos la hora
                    last_pred['fecha'] = get_timestamp(DAY_HOUR_FORMAT)
                    # Añadimos los valores al historial
                    st.session_state['historial'].append(last_pred)
                else:
                    # Mostramos advertencia
                    st.info("La evaluación ya se ha guardado.")
        else:
            st.info("Lanza una predicción para evaluar.")

    with tab_historial:
        # Creamos un dataframe con el historial guardado en sesión
        df = pd.DataFrame(st.session_state.get('historial'))
        # Sacamos una lista con los aciertos. Si ha acertado será 1 si no será 0
        aciertos = [float(d['pred'] == d['real']) for d in st.session_state.get('historial')]
        # Creamos el array de la suma acumulada
        aciertos = np.cumsum(aciertos)
        # Sacamos fechas
        fechas = [d['fecha'] for d in st.session_state.get('historial')]        

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(aciertos)), aciertos, label='acumulado aciertos')
        plt.xlabel('Tiempo')
        plt.title('Evolución de aciertos a lo Largo del Tiempo')
        plt.xticks(ticks=range(len(fechas)), labels=fechas)
        plt.yticks(ticks=range(len(aciertos)))
        st.pyplot(plt)

        st.dataframe(df, use_container_width=True, hide_index=True, column_order=['archivo', 'pred', 'conf', 'real', 'fecha'])
        # TODO Mostrar porcentaje de aciertos
        

    st.session_state

if __name__ == '__main__':
    main()
    