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

"""Script con la aplicación principal"""

# Librerías internas de python
from datetime import datetime
from io import BytesIO
import pytz
import time
from typing import Tuple
# Librerías de terceros
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
# Librerías propias del proyecto
from models.convnet_model import load_model
from streamlit_func import show_sidebar, config_page

# Constantes #
DAY_HOUR_FORMAT = """%d/%m/%y\n%H:%M"""
DAY_FORMAT = "%d/%m/%y"
COLOR_BLUE = '#213f99'

# Funciones auxiliares #
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
        Formato de tipo datetime para mostrar
        Por ejemplo: %d/%m/%y

    Returns
    -------
    str
        Devuelve momento actual del día
        en el formato especificado
    """
    return datetime.strftime(datetime.now(tz=pytz.timezone('Europe/Madrid')),format=formato)

# Validaciones #
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
        True si el nombre del archivo está en historial
        False en caso contrario
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

# Función principal #
def main() -> None:
    """Entry point de la app"""

    # Configuración de la app
    config_page()
    # Mostramos la Sidebar que hemos configurado en streamlit_func
    show_sidebar()
    # Título y descripción de la app
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
    # Definimos las 5 tabs que tendrá nuestra app
    tab_cargar_imagen, tab_ver_digito, tab_predecir, \
        tab_evaluar, tab_estadisticas = st.tabs(['Cargar imagen', 'Ver dígito', 
                                        'Predecir', 'Evaluar', 'Ver estadísticas'])
    ################
    ## TAB Cargar ##
    ################
    with tab_cargar_imagen:
        st.write('''Carga tu imagen con el dígito dibujado. 
            Recuerda que debe ser una imagen de 28x28 píxeles.<br>El dígito debe estar
            dibujado en blanco sobre color negro.''', unsafe_allow_html=True)
        
        imagen_bruta = st.file_uploader('Sube tu dígito', type=["png","tif","jpg","bmp","jpeg"], on_change=reset_predictions)
        # Si hay imagen cargada, convertimos en array la imagen y realizamos las validaciones de la imagen
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
            # Si la imagen es válida guardamos en sesión el nombre del archivo y mostramos un mensaje de éxito
            st.session_state['imagen_cargada_y_validada'] = imagen_bruta.name
            # Este mensaje solo se mostrará si hay una imagen cargada y si la imagen está validada
            st.success('Imagen cargada correctamente.')

    ################
    ## TAB Dígito ##
    ################        
    with tab_ver_digito:
        # Verificamos que tengamos una imagen cargada y validada en sesión
        if nombre_archivo:=st.session_state.get('imagen_cargada_y_validada'):
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(img_array, cmap="gray")
            ax.axis('off')
            ax.set_title(nombre_archivo, fontsize=5)
            st.pyplot(fig)
        else:
            st.info('Carga una imagen para visualizar.')

    ##################
    ## TAB Predecir ##
    ##################
    with tab_predecir:
        # Comprobamos si se ha lanzado una predicción y por tanto está almacenada en sesión.
        # Si tenemos una predicción la mostramos
        if (last_pred:=st.session_state.get('ultima_prediccion')) is not None:
            pred = last_pred['pred']
            conf = last_pred['conf']
            st.metric('Predicción del modelo', value=pred, delta=f"{'-' if conf < 0.7 else ''}{conf:.2%}")
        else:
            # Si no se ha lanzado una predicción, mostramos mecanismo para lanzarla
            # Verificamos que tengamos una imagen cargada y validada en sesión
            if nombre_imagen:=st.session_state.get('imagen_cargada_y_validada'):
                predecir = st.button(f'Predecir "{nombre_imagen}"')
                if predecir:
                    # Procesamos la imagen
                    img_processed = process_image(img_array)
                    # Lanzamos las predicciones
                    with st.spinner(text='Prediciendo dígito...'):
                        try:
                            pred, conf = predict(img_processed)
                            time.sleep(1)
                        except Exception as exc:
                            st.error(f'Se ha producido un error al predecir: {exc}')
                            st.stop()
                    # Si la confianza es menor del 70% ponemos un signo menos para que streamlit lo muestre
                    # en color rojo
                    st.metric('Predicción del modelo', value=pred, delta=f"{'-' if conf < 0.7 else ''}{conf:.2%}")
                    # Guardamos en sesión
                    st.session_state['ultima_prediccion'] = {
                        'pred': int(pred),
                        'conf': conf,
                        'archivo': nombre_imagen,
                    }
                    
            else:
                st.info('Carga una imagen para predecir.')

    #################
    ## TAB Evaluar ##
    #################
    with tab_evaluar:
        # Verificamos si hay una predicción lanzada y guardada en sesión
        if (last_pred:=st.session_state.get('ultima_prediccion')) is not None:
            # Posibilidad de contrastar con la realidad para almacenar porcentaje de aciertos
            st.subheader('¿ Ha acertado el modelo ?')
            digit = st.number_input('Marca el dígito que habías dibujado', min_value=0, max_value=9)
            guardar_pred = st.button('Guardar evaluación', help='Añade la evaluación al historial')
            # Si se pulsa el botón
            if guardar_pred:
                # Comprobamos que no hayamos guardado ya en sesión para no falsear las estadísticas
                if not pred_already_saved(last_pred['archivo']):
                    # Añadimos a ultima_prediccion la evaluación del usuario
                    last_pred['real'] = digit
                    # Añadimos la hora
                    last_pred['fecha'] = get_timestamp(DAY_HOUR_FORMAT)
                    # Añadimos los valores al historial
                    st.session_state['historial'].append(last_pred)
                    # Mostramos mensaje de éxito
                    st.success('Evaluación guardada correctamente.')
                else:
                    # Mostramos advertencia
                    st.info('La evaluación ya se ha guardado.')
        else:
            st.info('Lanza una predicción para evaluar.')

    ######################
    ## TAB Estadísticas ##
    ######################
    with tab_estadisticas:
        # Comprobamos que haya historial guardado en sesión
        if st.session_state.get('historial'):
            # Creamos un dataframe con el historial guardado en sesión
            df = pd.DataFrame(st.session_state.get('historial'))
            # Sacamos los aciertos comparando la variable pred y real
            df['acierto'] = df['pred'] == df['real']

            # Gráfico de evolución de confianzas
            st.line_chart(df, x='fecha', y='conf')

            # Ploteamos el acumulado de aciertos para representar la curva de acumulados
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(df)), np.cumsum(df['acierto']), label='acumulado aciertos', color=COLOR_BLUE)
            plt.xlabel('Tiempo')
            plt.ylabel('Aciertos acumulados')
            plt.title('Evolución de aciertos a lo Largo del Tiempo')
            plt.xticks(ticks=range(len(df)), labels=df['fecha'])
            plt.yticks(ticks=range(len(df)))
            plt.tight_layout()
            st.pyplot(plt)    

            # Agrupamos por el dígito real y calculamos el porcentaje de aciertos y el conteo de predicciones
            precision_y_conteo = df.groupby('real').agg({'acierto': 'mean', 'pred': 'count'})
            # Multiplicamos por 100 para obtener el porcentaje
            precision_y_conteo['acierto'] *= 100
            # Configuración del ancho de las barras
            bar_width = 0.35
            # Configuramos las posiciones de las barras
            indices = np.arange(len(precision_y_conteo))
            # Creamos el gráfico
            fig2, ax1 = plt.subplots()
            # Barras para el porcentaje de aciertos
            ax1.bar(indices - bar_width/2, precision_y_conteo['acierto'], bar_width, label='% de Aciertos', color='#213f99')
            # Creamos el segundo eje para el número de intentos
            ax2 = ax1.twinx()
            # Barras para el número de intentos de predicción
            ax2.bar(indices + bar_width/2, precision_y_conteo['pred'], bar_width, label='Número de Predicciones', color='orange')
            # Configuración de las etiquetas y títulos
            ax1.set_xlabel('Dígitos')
            ax1.set_ylabel('% de Aciertos', color=COLOR_BLUE)
            ax2.set_ylabel('Número de Predicciones', color='orange')
            # Configuramos el segundo eje y para usar solo números enteros
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.set_xticks(indices)
            ax1.set_xticklabels(precision_y_conteo.index)
            #ax1.legend(loc='best')
            #ax2.legend(loc='best')
            # Título del gráfico
            plt.title('Porcentaje de Aciertos y Número de Intentos por Dígito')
            # Mostramos el gráfico
            fig2.tight_layout()
            st.pyplot(fig2)

            # Mostramos el dataframe
            st.dataframe(df, use_container_width=True, hide_index=True, column_order=['archivo', 'pred', 'conf', 'real', 'fecha'])

        # Si no hay historial en sesión (lista vacía) mostramos mensaje de información 
        else:
            st.info('No hay estadísticas disponibles.')
        
if __name__ == '__main__':
    main()
    