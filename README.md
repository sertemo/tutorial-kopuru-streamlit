# Desarrollando y desplegando un modelo de reconocimiento de dígitos con Streamlit

En este artículo veremos cómo crear y desplegar una sencilla aplicación de reconocimiento de dígitos con **Streamlit**.

## Un poco sobre mi.
Mi nombre es Sergio Tejedor y soy un ingeniero industrial apasionado por el Machine Learning, la programación en Python y el desarrollo de aplicaciones. Actualmente soy director técnico de un grupo empresarial especializado en la laminación de perfiles en frío y fabricación de invernaderos industriales. Sin embargo, hace ya más de un año decidí dar el salto y ponerme a estudiar por mi cuenta programación. Escogí el lenguaje Python por su suave curva de aprendizaje y sobre todo porque es uno de los lenguajes más utilizados en la ciencia de datos y el Machine Learning. Al ir profundizando y descubriendo la gran flexibilidad de este lenguaje de programación, de todos las posibilidades, decidí centrarme en el campo del Machine Learning ya que es aquel que más sinergias podía tener con mi formación y profesión.

## ¿ Por qué Streamlit ?
Enseguida empecé a desarrollar aplicaciones y querer compartirlas con amigos. No tardé en descubrir el framework para Python [Streamlit](https://streamlit.io/). Siempre había encontrado dificultades a la hora de desplegar aplicaciones (y las sigo encontrando) y es precisamente ahí dónde destaca esta librería. Streamlit permite crear y desplegar aplicaciones web dinámicas en las que poder compartir y visualizar tus proyectos de ciencia de datos utilizando únicamente Python y sin necesidad de conocimiento profundo de tecnologías web. Mediante los widgets que tienen disponibles, se pueden cargar archivos, visualizar gráficos, dataframes y muchas cosas más.
Tiene una comunidad creciente de usuarios y desarrolladores y una buena documentación con ejemplos ilustrativos.

## Explicación de la app.
Las posibilidades con Streamlit son enormes pero hoy detallaremos cómo desarrollar una aplicación de reconocimiento de dígitos. La aplicación está disponible [aquí](https://tutorial-kopuru.streamlit.app/).
![Alt text](img/aplicacion_view.JPG)
Se le permite a un usuario cargar un dígito dibujado para que un modelo previamente entrenado lo intente reconocer. Asimismo, también mostrará algunas gráficas y estadísticas de las predicciones que se van haciendo.
La aplicación es multi página (2 en este caso) y está organizada en 5 pestañas:
### Cargar imagen
![Alt text](img/pesta%C3%B1a_cargar_imagen.JPG)
En esta pestaña el usuario podrá cargar la imagen con su dígito.
### Ver dígito
![Alt text](img/pesta%C3%B1a_ver_digito.JPG)
Se muestra el dígito dibujado por el usuario.
### Predecir
![Alt text](img/pesta%C3%B1a_predecir.JPG)
Mediante un botón se lanza la predicción del modelo que se mostrará más abajo.
## Evaluar
![Alt text](img/pesta%C3%B1a_evaluar.JPG)
En esta pestaña se evalúa el modelo contrastándolo con el dígito real que se pretendía dibujar y se da la posibilidad de guardar dicha evaluación
### Ver estadísticas
![Alt text](img/pesta%C3%B1a_estadisticas.JPG)
Con todas las evaluaciones guardadas se realizan algunas gráficas estadísticas



Guión:
1. Contar algo sobre ti (nos gusta que os vendáis un poco 😊 y que os conozcan)
2. Contar por qué usas esta aplicación o las ventajas que has visto en ella (muy breve, queremos que la importancia no este en la App, si no en lo que puedes hacer a través de ella)
3. Desarrollar la idea que quieres poner en marcha, por ejemplo, la de detección de dígitos e imágenes.
4. Contar el paso a paso de cómo se desarrolla.
5. Incluir código sólo en esos apartados donde es más fácil matizar algún detalle si se tiene el código delante. Y en el resto de pasos, remitir al git, donde lo tienes completo.