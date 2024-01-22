# Artículo para el Blog de Kopuru

## Posible título
- Desarrollando y desplegando un modelo de reconocimiento de dígitos con Streamlit

## Esquema

### Introducción
- Qué es Streamlit. Ventajas y limitaciones. Por qué para proyectos de Data Science.

#### Descripción del proyecto
- Entrenaremos un modelo convolucional (ConvNet) sencillo capaz de reconocer dígitos sencillos del 0-9 en imágenes de 28x28 píxeles. Se usará la API de **Keras**, el dataset **MNIST** y **Streamlit** como plataforma para evaluar imágenes generadas por el usuario.
- Paquetes, módulos o programas que se utilizarán:
    - **poetry** como administrador de dependencias y entorno virtual. [ver doc](https://python-poetry.org/docs/basic-usage/)
    - **git** como administrador de versiones. [ver doc](https://git-scm.com/docs)
    - **Keras** como librería para deep learning. [ver doc](https://keras.io/about/)
    - **Vscode** como editor de código. [enlace](https://code.visualstudio.com/)
    - **Python** en su versión 3.11. [enlace](https://www.python.org/downloads/)
    - **Windows 10**
    - **Streamlit** como framework de aplicación web. [ver doc](https://docs.streamlit.io/)

### Parte 1: Preparación y entrenamiento del modelo

#### Configuración inicial: Entorno virtual y git
- Descargar e instalar git [enlace](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Instalar **poetry**
    pip install poetry
- Iniciar poetry
    poetry init
- Configuración in-project
    poetry config virtualenvs.in-project true
- Instalar el entorno virtual
    poetry install
- Añadir librerías necesarias
    poetry add streamlit keras ...
- Activar entorno virtual
    poetry shell

- Inicializar **git**
    git init
- Hacer *commits* para guardar nuestro avance
    git add .
    git commit -m "Inicializado poetry y git"

#### Estructura del proyecto. Jerarquía de archivos

#### Dataset MNIST 
- Cargar el dataset, visualizar los datos
- Preprocesamiento

#### Construcción del modelo ConvNet 
(recomendado en [Google Colab](https://colab.research.google.com/?hl=es))
- Arquitectura elegida
- Functional API
- Loss, métricas, compilación y entrenamiento
- Evaluación de la precisión
- Guardado del modelo en disco

### Parte 2: Creación de la aplicación con Streamlit

#### Introducción
- Explicación de Streamlit: widgets etc
- Registrarse en Streamlit (con Google por ejemplo)

#### Diseño de la Interfaz de Usuario
- Dibujo esquemático de las diferentes secciones de la app:
    - Configuración de la app (.streamlit)
    - Título
    - Carga del input usuario (imagen)
    - Visualización de la imagen del usuario
    - Visualización de Parámetros del modelo entrenado
    - Visualización de las predicciones y % de confianza

#### Integración del modelo en el código
- Creación del código de la aplicación e integración del modelo entrenado
- Archivo *main.py*
- Correr la aplicación en local
    streamlit run main.py
- Validaciones varias (imagen del usuario)
- Algunos trucos para *beautify* la aplicación inyectando HTML

#### Despliegue con GitHub
- Registrarse en GitHub
- Crear un repositorio
- Creación del archivo requirements.txt
    poetry export -f requirements.txt --output requirements.txt
- Commits y push
    git add .
    git commit -m "aplicacion terminada lista para desplegar"
    git remote add origin git@github.com:<usuario>/<nombre-aplicacion>.git
- Crear nueva app en Streamlit (Deploy an app)






