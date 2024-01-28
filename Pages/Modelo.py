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

"""Script que recoge el código relacionado con la visualización de las
características del modelo entrenado"""

import streamlit as st

from streamlit_func import show_sidebar

def main() -> None:
    """Entry point de la app"""

    # Configuración de la app
    st.set_page_config(
        page_title=f"Reconocimiento de dígitos",
        page_icon="👁️", 
        layout="wide",
        initial_sidebar_state="auto",
    )

    show_sidebar()
        
    # TODO Matriz zde Confusión
    # TODO Parámetros del modelo
        
if __name__ == '__main__':
    main()