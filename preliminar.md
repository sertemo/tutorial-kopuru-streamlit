# Artículo/Tutorial para el Blog de Kopuru 🧠

## Posible título
- Desarrollando y desplegando un modelo de reconocimiento de dígitos con Streamlit

## Esquema

### Introducción
- Qué es Streamlit. Ventajas y limitaciones. Por qué para proyectos de Data Science.

#### Descripción del proyecto
- Entrenaremos un modelo convolucional (ConvNet) sencillo capaz de reconocer dígitos sencillos del 0-9 en imágenes de 28x28 píxeles. Se usará la API de **Keras**, el dataset **MNIST** y **Streamlit** como plataforma para evaluar imágenes generadas por el usuario.
- Paquetes, módulos o programas que se utilizarán:
    - **conda** para entorno virtual. [ver doc](https://python-poetry.org/docs/basic-usage/)
    - **git** como administrador de versiones. [ver doc](https://git-scm.com/docs)
    - **Tensorflow** y **Keras** como librería para deep learning. [ver doc](https://keras.io/about/)
    - **Vscode** como editor de código. [enlace](https://code.visualstudio.com/)
    - **Python** en su versión 3.9.6.
    - **Windows 10**
    - **Streamlit** como framework de aplicación web. [ver doc](https://docs.streamlit.io/)
    - **Streamlit Community Cloud** Para el despliegue. [enlace](https://streamlit.io/cloud)

### Parte 1: Preparación y entrenamiento del modelo

#### Configuración inicial: Entorno virtual y git
- Descargar e instalar git: [enlace](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Instalar **conda** si no está instalado: [enlace](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)

- Crear un entorno virtual llamado por ejemplo **tensorflow**
```sh
$ conda create --name tensorflow python=3.9
```
- Ver entornos virtuales disponibles
```sh
$ conda info --envs
```
- Activar el entorno virtual
```sh
$ conda activate tensorflow
```
- Añadir librerías necesarias
```sh
$ pip install tensorflow streamlit keras 
```
- Ver librerías instaladas
```sh
$ conda list 
```
- Crear proyecto en VsCode
- Seleccionar Interprete del proyecto (Ctrl+shift+P)
- Inicializar **git** y renombrar rama principal a **main**.
```sh
$ git init
$ git branch -m main
```
- Opcional pero recomendable: cambiar la configuración para que siempre se inicialice como main y no master
```sh
$ git config --global init.defaultBranch main
```

- Hacer *commits* para guardar nuestro avance
```sh
$ git add .
$ git commit -m "Inicializado entorno virtual y git"
```

#### Estructura del proyecto. Jerarquía de archivos

#### Dataset MNIST 
- Cargar el dataset, visualizar los datos
- Preprocesamiento

#### Construcción del modelo ConvNet 
(recomendado en [Google Colab](https://colab.research.google.com/?hl=es))
- Arquitectura elegida
- Functional API de keras
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
- App multi-página
- Archivo *main.py*
- Ejecutar la aplicación en local e ir viendo cambios
```sh
$ streamlit run main.py
```

- Validaciones varias (input del usuario)
- Algunos trucos para *beautify* la aplicación inyectando HTML

#### Despliegue con GitHub
- Registrarse en GitHub
- Crear un repositorio
- Creación del archivo requirements.txt
```sh
$ conda list -e | grep -v "^#" | awk -F'=' '{print $1 "==" $2}' > requirements.txt
```

- Commits y push
```sh
git add .
git commit -m "aplicacion terminada lista para desplegar"
git remote add origin git@github.com:<usuario>/<nombre-aplicacion>.git
git push -u origin main
```

- Crear nueva app en Streamlit (Deploy an app)
    - Seleccionar el repositorio






