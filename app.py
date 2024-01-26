
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
    st.write(':gray[Modelo]')

st.title('Reconocimiento de dígitos')
st.subheader('Una App para Kopuru')
st.write('''Con esta app serás capaz de evaluar un modelo convolucional entrenado 
            con el dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
            Dibuja un dígito en color blanco sobre fondo negro en una imagen de 28x28 píxeles
            y súbela. Pulsa sobre el botón **predecir** y comprueba si el modelo ha sido
            capaz de averiguar el dígito que habías dibujado.''')

tab1, tab2, tab3 = st.tabs(['Cargar Imagen', 'Ver dígito', 'Ver Predicciones'])

with tab1:
    st.write('''Carga tu imagen con el dígito dibujado. 
        Recuerda que debe ser una imagen de 28x28 píxeles. El dígito debe estar
        dibujado en blanco sobre color negro.''')
    
    imagen = st.file_uploader('Sube tu dígito', type=["png","tif","jpg","bmp","jpeg"])