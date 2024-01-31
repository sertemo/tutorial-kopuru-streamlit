# Desarrollando y desplegando un modelo de reconocimiento de dígitos con Streamlit

En este artículo veremos cómo crear y desplegar una sencilla aplicación de reconocimiento de dígitos con **Streamlit**.

## Un poco sobre mi.
Mi nombre es Sergio Tejedor y soy un ingeniero industrial apasionado por el **Machine Learning**, la programación en **Python** y el desarrollo de aplicaciones. Actualmente soy director técnico de un grupo empresarial especializado en la laminación de perfiles en frío y fabricación de invernaderos industriales. Sin embargo, hace ya más de un año decidí dar el salto y ponerme a estudiar por mi cuenta programación. 

Escogí **Python** como lenguaje por su suave curva de aprendizaje y sobre todo porque es uno de los lenguajes más utilizados en la ciencia de datos y el **Machine Learning**. Al profundizar y descubrir la gran flexibilidad de este lenguaje de programación, de entre todas sus posibilidades, decidí centrarme en el campo del **Machine Learning** ya que es aquel que más sinergias podía tener con mi formación y profesión.

## ¿ Por qué Streamlit ?
Enseguida empecé a desarrollar aplicaciones y querer compartirlas con amigos. No tardé en descubrir el framework para Python [**Streamlit**](https://streamlit.io/). Siempre había encontrado dificultades a la hora de desplegar aplicaciones (y las sigo encontrando) y es precisamente ahí dónde destaca esta librería. Streamlit permite crear y desplegar aplicaciones web dinámicas en las que poder compartir y visualizar tus proyectos de ciencia de datos utilizando únicamente Python y sin necesidad de conocimiento profundo de tecnologías web. Mediante los widgets que tienen disponibles, se pueden cargar archivos, visualizar gráficos, dataframes y muchas cosas más.
Tiene una comunidad creciente de usuarios y desarrolladores y una buena [documentación](https://docs.streamlit.io/library/api-reference) con ejemplos ilustrativos.

## Descripción de la app.
Las posibilidades con **Streamlit** son casi ilimitadas pero hoy abordaremos el desarrollo de una aplicación de reconocimiento de dígitos. La aplicación está disponible [aquí](https://tutorial-kopuru.streamlit.app/).
![Alt text](img/aplicacion_view.JPG)
Se le permite a un usuario cargar un dígito dibujado para que un modelo previamente entrenado lo intente reconocer. Asimismo, también mostrará algunas gráficas y estadísticas de las predicciones que se van haciendo.
La aplicación es multi página (2 en este caso).
La página principal, **Aplicación**, está organizada en **5 pestañas**:
### Cargar imagen
![Alt text](img/pesta%C3%B1a_cargar_imagen.JPG)
- En esta pestaña el usuario podrá cargar la imagen con su dígito.
### Ver dígito
![Alt text](img/pesta%C3%B1a_ver_digito.JPG)
- Se muestra el dígito dibujado por el usuario.
### Predecir
![Alt text](img/pesta%C3%B1a_predecir.JPG)
- Mediante un botón se lanza la predicción del modelo que se mostrará más abajo.
### Evaluar
![Alt text](img/pesta%C3%B1a_evaluar.JPG)
- En esta pestaña se evalúa el modelo contrastándolo con el dígito real que se pretendía dibujar y se da la posibilidad de guardar dicha evaluación
### Ver estadísticas
![Alt text](img/pesta%C3%B1a_estadisticas.JPG)
- Con todas las evaluaciones guardadas se realizan algunas gráficas estadísticas

La única página secundaria, **Modelo**, tiene detalles del modelo entrenado. Al acceder a ella se muestra en forma de *'stream'* las capas y parámetros del modelo.

## Desarrollo de la app.
Crea un nuevo proyecto en tu IDE favorito. Personalmente uso **Visual Studio Code** para desarrollar mis aplicaciones ya que lo encuentro fácil de usar y muy personalizable. La gran cantidad de extensiones disponibles facilitan mucho el desarrollo del código.

Crearemos el script **Aplicacion.py** que recogerá el código de la página principal y la carpeta **pages** con el archivo **1_Modelo.py** en su interior. Las aplicaciones multipágina en streamlit se configuran de este modo; una página inicial o principal en la ruta raiz del proyecto y todas las páginas secundarias deberán ir dentro de una carpeta **pages**. Es altamente recomendable nombrar los scripts de las páginas secundarias con los prefijos 1_xx, 2_xx, 3_xx etc para que streamlit pueda reconocerlos correctamente.

El siguiente paso es configurar un entorno virtual e instalar las dependencias necesarias. Para proyectos en los que utilizo algún tipo de modelo de deep learning con tensorflow** tengo un entorno virtual llamado **tensorflow** creado con **conda** con todos los paquetes necesarios. Puedes crear el entorno virtual usando Python en su versión 3.9 con el siguiente comando en la terminal:
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

Para poder acceder a todas las funcionalidades necesitamos importar **streamlit** con el alias **st** (esto es por convención). Seguidamente escribiremos la función **main** y en ella configuraremos algunos parámetros de la aplicación como el **título**, el **icono**, el tipo de **layout** y la configuración inicial de la **barra lateral**.
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









----------------------------------------
----------------------------------------
Guión:
1. Contar algo sobre ti (nos gusta que os vendáis un poco 😊 y que os conozcan)
2. Contar por qué usas esta aplicación o las ventajas que has visto en ella (muy breve, queremos que la importancia no este en la App, si no en lo que puedes hacer a través de ella)
3. Desarrollar la idea que quieres poner en marcha, por ejemplo, la de detección de dígitos e imágenes.
4. Contar el paso a paso de cómo se desarrolla.
5. Incluir código sólo en esos apartados donde es más fácil matizar algún detalle si se tiene el código delante. Y en el resto de pasos, remitir al git, donde lo tienes completo.