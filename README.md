# Desarrollando y desplegando un modelo de reconocimiento de dígitos con Streamlit

En este artículo veremos cómo crear y desplegar una sencilla aplicación de reconocimiento de dígitos con **Streamlit**.

## Un poco sobre mi.
Mi nombre es Sergio Tejedor y soy un ingeniero industrial apasionado por el **Machine Learning**, la programación en **Python** y el desarrollo de aplicaciones. Actualmente soy director técnico de un grupo empresarial especializado en la laminación de perfiles en frío y fabricación de invernaderos industriales. Hace ya algo más de un año decidí dar el salto y ponerme a estudiar por mi cuenta programación. 

Escogí **Python** como lenguaje por su suave curva de aprendizaje y por ser uno de los lenguajes más utilizados en la ciencia de datos y el **Machine Learning**. Al profundizar y descubrir la gran flexibilidad de este lenguaje de programación, de entre todas sus posibilidades, decidí centrarme en el campo del **Machine Learning** ya que es aquel que más sinergias podía tener con mi formación y profesión.

## ¿ Por qué Streamlit ?
Enseguida empecé a desarrollar aplicaciones y querer compartirlas con amigos. No tardé en descubrir el framework para Python [**Streamlit**](https://streamlit.io/). Siempre había encontrado dificultades a la hora de desplegar aplicaciones (y las sigo encontrando) y es precisamente este uno de los puntos fuertes de esta librería. **Streamlit** permite crear y desplegar aplicaciones web dinámicas en las que poder compartir y visualizar tus proyectos de ciencia de datos utilizando únicamente Python y sin necesidad de conocimiento profundo de tecnologías web. De forma rápida y muy sencilla. Mediante los widgets disponibles, se pueden cargar *archivos*, *visualizar gráficos*, *dataframes* y muchas cosas más.
Tiene una comunidad creciente de usuarios y desarrolladores y una buena [documentación](https://docs.streamlit.io/library/api-reference) con ejemplos ilustrativos.

## Descripción de la app.
Las posibilidades con **Streamlit** son casi ilimitadas pero hoy abordaremos el desarrollo de una aplicación de reconocimiento de dígitos. La aplicación está disponible [aquí](https://tutorial-kopuru.streamlit.app/).
![Alt text](img/aplicacion_view.JPG)
La aplicación ofrece además la posibilidad de evaluar las predicciones del modelo y mostrar una serie de estadísticas.

En definitiva, creo que es un buen ejemplo de muchas de las posibilidades que ofrece **Streamlit** a la hora de diseñar una aplicación web.

A nivel de estructura, la aplicación cuenta con 2 páginas:

La página principal, **Aplicación**, está organizada en **5 pestañas**:

### Cargar imagen
![Alt text](img/pesta%C3%B1a_cargar_imagen.JPG)
- En esta pestaña es donde el usuario podrá cargar su dígito.

### Ver dígito
![Alt text](img/pesta%C3%B1a_ver_digito.JPG)
- Aquí se mostrará una representación de la imagen del usuario.

### Predecir
![Alt text](img/pesta%C3%B1a_predecir.JPG)
- Mediante un botón se lanza la predicción del modelo que se mostrará más abajo.

### Evaluar
![Alt text](img/pesta%C3%B1a_evaluar.JPG)
- En esta pestaña se evalúa el modelo contrastándolo con el dígito real que se pretendía dibujar y se da la posibilidad de guardar dicha evaluación.

### Ver estadísticas
![Alt text](img/pesta%C3%B1a_estadisticas.JPG)
- Con todas las evaluaciones guardadas se realizan algunas gráficas estadísticas.

La única página secundaria, **Modelo**, tiene detalles del modelo entrenado. Al acceder a ella se muestran en forma de *'stream'* las capas y parámetros del modelo.

## Desarrollo de la app.
Crea un nuevo proyecto en tu editor de código favorito. Personalmente uso [**Visual Studio Code**](https://code.visualstudio.com/) para desarrollar mis aplicaciones ya que lo encuentro fácil de usar y muy personalizable. La gran cantidad de extensiones disponibles facilitan mucho el desarrollo del código.

Crearemos el script **Aplicacion.py** que recogerá el código de la página principal y la carpeta **pages** con el archivo **1_Modelo.py** en su interior. Las aplicaciones multipágina en streamlit se configuran de este modo; una página inicial o principal en la ruta raiz del proyecto y todas las páginas secundarias deberán ir dentro de una carpeta **pages**. Es recomendable nombrar los scripts de las páginas secundarias con los prefijos 1_xx, 2_xx, 3_xx etc para que streamlit pueda reconocerlos correctamente.

El siguiente paso es configurar un entorno virtual e instalar las dependencias necesarias. Para proyectos en los que utilizo algún tipo de modelo de deep learning con tensorflow tengo un entorno virtual llamado **tensorflow** creado con **conda** con todos los paquetes necesarios. Puedes crear el entorno virtual usando Python en su versión 3.9 con el siguiente comando en la terminal:
 ```sh
$ conda create --name tensorflow python=3.9
```
No olvides activar el entorno virtual
```sh
$ conda activate tensorflow
```
Para este proyecto concretamente instalaremos las siguientes dependencias:
```sh
$ pip install streamlit tensorflow numpy pandas Pillow matplotlib
```
Para realizar el despliegue de la aplicación una vez hayamos terminado su desarrollo, será necesario tener una cuenta en [**GitHub**](https://github.com/) y aunque pueda hacerse al final, suele ser aconsejable iniciar **git** en el proyecto e ir guardando los avances de la aplicación en tu repositorio de GitHub. Ya sabes, por lo que pueda pasar.

**Streamlit** permite configurar algunos temas para toda la aplicación. Esto se realiza con un archivo **config.toml** que debe insertarse en una carpeta **.streamlit** en la rama principal de tu proyecto.

Algunas [personalizaciones](https://docs.streamlit.io/library/advanced-features/theming) posibles son: el color del texto, los colores de fondo, colores de acento y tipo de letra.
```js
[theme]
# backgroundColor="#EFF6EE"
# secondaryBackgroundColor="#aab0bd"

primaryColor="#f2a416"
base="dark" # o "light"
textColor="#EEEDEB"
font="sans serif"
```
Hasta ahora, esta es la jerarquía de archivos de nuestro proyecto:
```sh
Carpeta Principal del proyecto
├── .streamlit
│   └── config.toml
├── .git
├── pages
│   └── 1_Modelo.py
└── Aplicacion.py
```
### Script principal. Aplicacion.py
Es hora de empezar a picar el código de nuestro archivo principal: **Aplicacion.py**. Para ver el código completo de la aplicación comentado, puedes echar un vistazo a mi [repositorio](https://github.com/sertemo/tutorial-kopuru-streamlit) en github.

Casi siempre estructuro el script en varios bloques ([PEP8](https://peps.python.org/pep-0008/) para buenas prácticas en Python):
1. Importaciones, ordenadas de la siguiente manera:
    1. Librerías internas
    2. Librerías de terceros (*pip install*)
    3. Librerías o módulos de la aplicación
2. Constantes usadas en el script
3. Funciones auxiliares usadas en el script
4. Función principal **main**
5. Entry point de la app con:
    ```py
    if __name__ == '__main__':
        main()
    ```

Para poder acceder a todas las funcionalidades necesitamos importar **streamlit** con el alias **st** (esto es por convención). Aprovechamos también para importar el resto de librerías que vamos a ir necesitando:
```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
```

Seguidamente escribiremos la función **main** y en ella configuraremos algunos parámetros de la aplicación como el **título**, el **icono**, el tipo de **layout** y la configuración inicial de la **barra lateral**.
```py
st.set_page_config(
        page_title=f"Reconocimiento de dígitos",
        page_icon="👁️", 
        layout="centered",
        initial_sidebar_state="auto",
    )
```

**Streamlit** nos permite ir viendo el progreso de la aplicación en un servidor local de forma sencilla. Para ello simplemente debemos escribir lo siguiente en nuestra terminal:
```sh
$ streamlit run Aplicacion.py
```

Cada vez que avancemos en el código y guardemos, streamlit actualizará la visualización.

Un detalle muy importante que debemos tener en cuenta siempre que diseñemos una aplicación con **Streamlit** es que **en cada interacción del usuario con cualquier elemento de la aplicación**, los scripts se ejecutarán de arriba a abajo siempre. Esto es fundamental tenerlo claro cuando implementemos nuestra lógica, sobre todo a la hora de asignar variables. No te preocupes, **Streamlit** pone a disposición de los desarrolladores unas variables de sesión que veremos más adelante.

Debería mostrarse una [barra lateral](https://docs.streamlit.io/library/api-reference/layout/st.sidebar) con unas opciones seleccionables: **Aplicacion** y **Modelo**.

Para agregar más elementos a la barra lateral podemos hacerlo con la notación **with**:
```py
with st.sidebar:
    # Todo lo que está en el interior
    # del bloque 'with' aparecerá en la barra lateral.
```

Podemos agregar nuestra firma, una imagen o lo que nos plazca. **Streamlit** tiene varias posibilidades de [escritura](https://docs.streamlit.io/library/api-reference/text):
- **st.markdown()** : permite agregar notación **markdown** y HTML si pasamos el argumento **unsafe_allow_html**.
- **st.title()**
- **st.header()**
- **st.subheader()**
- **st.caption()** : escribe texto en letra pequeña.
- **st.code()** : escribe un bloque de código.
- **st.text()** : escribe texto plano.
- **st.latex()** : escribe texto formateado para expresiones matemáticas.

Podemos cargar una imagen de la misma manera, con la función **st.image()** y especificando la ruta del archivo.

Una vez hayamos terminado con el título y la descripción de la aplicación, incorporaremos las diferentes pestañas tal y como hemos descrito en el apartado **Descripción de la app.**

Para agregar una [pestaña](https://docs.streamlit.io/library/api-reference/layout/st.tabs), se sigue una lógica similar a la de la barra lateral. La función **st.tabs()** requiere de una lista como argumento que representa el nombre de las etiquetas de las pestañas. La función devuelve una tupla de igual longitud que la lista de nombres pasados.
```py
# Definimos las 5 tabs que tendrá nuestra app
tab_cargar_imagen, tab_ver_digito, tab_predecir, \
    tab_evaluar, tab_estadisticas = st.tabs(['Cargar imagen', 'Ver dígito', 
                                    'Predecir', 'Evaluar', 'Ver estadísticas'])
```

Los objetos devueltos por la función **st.tabs()** pueden usarse con la notación **with** para incorporar elementos dentro de cada pestaña:
```py
with tab_cargar_imagen:
    st.write('''Carga tu imagen con el dígito dibujado. 
        Recuerda que debe ser una imagen de 28x28 píxeles.<br>El dígito debe estar
        dibujado en blanco sobre color negro.''', unsafe_allow_html=True)
```
#### Pestaña Cargar imagen.
En esta primera pestaña vamos a dar la posibilidad al usuario de cargar una imagen. Realizaremos las siguientes etapas:
- Guardar la imagen del usuario en una variable
- Transformar la imagen en un array de **numpy**
- Verificar que la imagen cumpla los parámetros requeridos para pasarla por nuestro modelo
- Guardamos el nombre del archivo en sesión y mostramos mensaje de éxito

Para poder guardar el contenido de un archivo en una variable, streamlit pone a nuestra disposición la función **st.file_uploader()**. Al llamar a esta función, streamlit mostrará el widget de carga de archivos en la web.
```py
imagen_bruta = st.file_uploader(label='Sube tu dígito', type=["png","tif","jpg","bmp","jpeg"], on_change=reset_predictions)
```

Esta función nos devuelve o **None** si no hay ninguna imagen o un objeto de **UploadedFile**. El resto de lógica dentro de esta pestaña solo deberá ejecutarse **si** el usuario ha cargado una imagen. Recuerda que cada interacción del usuario con la aplicación ejecuta todos los scripts de arriba a abajo. Añadimos un bloque condicional para agrupar el resto del código de esta pestaña:
```py
if imagen_bruta is not None:
    # TODO: Transformar en array
    # TODO: Validar la imagen
    # TODO: Si no es válida detener la ejecución de la aplicación
    # TODO: Guardar nombre del archivo en sesión
    # TODO: Mostrar mensaje de éxito.
```

Para transformar los datos de la imagen en un array primero tenemos que leer el contenido del objeto **UploadedFile** que nos lo devolverá en bytes. La librería [Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html) es muy utilizada en Python para el tratamiento de imágenes y el método **open** dentro del módulo **Image** nos permite abrir la imagen pero antes es necesario envolver los bytes devueltos por el objeto UploadedFile con la clase **BytesIO** para tratar dichos bytes como si de un archivo se trataran. El código queda así:
```py
img_array = np.array(Image.open(BytesIO(imagen_bruta.read())))
```

**BytesIO** se importa de la librería **io** que viene incluida con Python.

Si la imagen cargada por el usuario no pasa nuestras funciones de validación, es apropiado mostrar un mensaje de error al usuario y detener la aplicación, de lo contrario podríamos encontrarnos con múltiples errores.
```py
if not valid_img:
                st.error(error_msg)
                st.stop() # Lo que viene después del stop no se ejecutará.
```

**Streamlit** también permite volver a ejecutar la aplicación con la función **st.rerun()**.

Para poder guardar información de manera persistente y que sobreviva a las interacciones del usuario con la aplicación, streamlit pone a nuestra disposición [**st.session_state**](https://docs.streamlit.io/library/api-reference/session-state).

**st.session_state** es un diccionario (un objeto diccionario clásico de Python) de sesión en el que podemos escribir y extraer la información que queramos. Este diccionario no se reescribe cada vez que se ejecuta el código pero sí se reinicia cuando volvemos a cargar la sesión (cuando volvemos a cargar la aplicación). Si quisiéramos que la información fuera persistente entre sesiones tendríamos que almacenarla en una base de datos. Para nuestra aplicación de hoy, st.session_state será suficiente.

Suele ser de gran ayuda también mostrar el contenido de st.session_state a medida que se desarrolla la aplicación. Para ello puedes escribir lo siguiente al principio o final de tu script:
```py
st.session_state
```

En nuestro caso guardaremos el nombre del archivo con la clave *imagen_cargada_y_validada*.
```py
# Si la imagen es válida guardamos en sesión el nombre del archivo y mostramos un mensaje de éxito
st.session_state['imagen_cargada_y_validada'] = imagen_bruta.name
# Este mensaje solo se mostrará si hay una imagen cargada y si la imagen está validada
st.success('Imagen cargada correctamente.')
```

**Streamlit** permite usar 4 tipos de [mensaje de estado](https://docs.streamlit.io/library/api-reference/status):
- **st.error()** : muestra el mensaje sobre fondo rojo.
- **st.warning()** : muestra el mensaje sobre fondo amarillo.
- **st.info()** : muestra el mensaje sobre fondo azul.
- **st.success()** : muestra el mensaje sobre fondo verde.

Todos ellos tienen la posibilidad de incluir un icono en el mensaje.

#### Pestaña Ver dígito.
Una vez terminada la implementación de la primera pestaña podemos pasar a la segunda. En esta pestaña simplemente mostraremos la imagen cargada por el usuario para poder visualizar el dígito.

**Streamlit** incorpora funciones propias para visualizar varios tipos de [gráficos](https://docs.streamlit.io/library/api-reference/charts) y además también ofrece la posibilidad de visualizar gráficos realizados con **matplotlib**. Los gráficos propios de streamlit son interactivos mientras que aquellos realizados con librerías como matplotlib se visualizan como imágenes.

Mostraremos el dígito del usuario con el siguiente código:
```py
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
```

De nuevo, es importante tener en cuenta que cada interacción del usuario con los elementos de la aplicación hará que se ejecute todo el código. Por ello, para evitar errores, verificamos primero que exista una imagen cargada y haya sido validada. Comprobamos si en sesión existe el campo **imagen_cargada_y_validada** que hemos guardado previamente en la pestaña anterior tras realizar las validaciones. De ser así, mostramos el gráfico.

Para visualizar un gráfico de matplotlib en streamlit basta con pasarle el objeto **Figure** devuelto por **subplots** a la función **st.pyplot()**.

![Alt text](img/ver_digito_2.JPG)

#### Pestaña Predecir.

#### Pestaña Evaluar.

#### Pestaña Estadísticas.











