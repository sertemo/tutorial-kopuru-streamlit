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

import streamlit as st

def imagen_con_enlace(url_imagen, url_enlace, 
                    alt_text="Imagen", 
                    max_width:int=100, 
                    centrar:bool=False, 
                    radio_borde:int=15,
                    ) -> None:
    """Muestra una imagen que es también un hipervínculo en Streamlit con bordes redondeados.

    Args:
    url_imagen (str): URL de la imagen a mostrar.
    url_enlace (str): URL a la que el enlace de la imagen debe dirigir.
    alt_text (str): Texto alternativo para la imagen.
    max_width (int): Ancho máximo de la imagen como porcentaje.
    centrar (bool): Si es verdadero, centra la imagen.
    radio_borde (int): Radio del borde redondeado en píxeles.
    """    
    html = f'<a href="{url_enlace}" target="_blank"><img src="{url_imagen}" alt="{alt_text}" style="max-width:{max_width}%; height:auto; border-radius:{radio_borde}px;"></a>'
    if centrar:
        html = f"""
                    <div style='text-align: center'>
                        {html}
                    </div>
                    """
    st.markdown(html, unsafe_allow_html=True)