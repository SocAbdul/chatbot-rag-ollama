# 🦙 Aplicación RAG con Streamlit y Ollama (Modelo phi3)

Descripción General
Esta guía te muestra cómo construir una aplicación de Generación Aumentada por Recuperación (RAG) usando:

Streamlit para la interfaz web.
Ollama para ejecutar un LLM local (phi3, un modelo ligero).
LangChain para la recuperación de documentos y el encadenamiento de procesos.
Chroma como la base de datos vectorial local.
¡Puedes subir un archivo PDF o de texto y hacer preguntas! Tu IA local responderá usando la información de tu documento.

### Prerrequisitos

Windows 10/11 (también funciona en Linux/Mac).
Python 3.9+.
Ollama instalado y en ejecución.
Conexión a internet para la configuración inicial.
Instalación

1. Instalar dependencias de Python

```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install streamlit langchain chromadb pypdf ollama requests
```

2. Instalar e iniciar Ollama

Descarga Ollama desde: https://ollama.com/download

Después de la instalación, abre una terminal y ejecuta:

```bash
ollama run phi3
```
Esto descargará e iniciará el modelo ligero phi3 localmente.

Uso
Guarda el código de la aplicación (que se encuentra más abajo) como rag_ollama_app.py.
Ejecuta la aplicación de Streamlit:
Bash

streamlit run rag_ollama_app.py
Abre tu navegador en http://localhost:8501.
¡Sube tu documento y haz preguntas!
Código con Anotaciones
Python

# Importaciones: Librerías principales para la app
```bash
import streamlit as st 
```
# Para la interfaz web
```bash
import os
import tempfile                         # Para el manejo de archivos temporales
import requests                         # Para verificar si el servidor de Ollama está en ejecución

# --- LangChain y dependencias para el pipeline RAG ---
from langchain.document_loaders import PyPDFLoader, TextLoader    # Para cargar archivos PDF/TXT
from langchain.embeddings import OllamaEmbeddings                 # Para vectorizar texto usando modelos de Ollama
from langchain.vectorstores import Chroma                         # Para almacenar y buscar los vectores (embeddings)
from langchain.llms import Ollama                                 # Para conectarse al LLM de Ollama
from langchain.chains import RetrievalQA                          # Cadena de LangChain para RAG
```

# . Configuración de la página de Streamlit 
```bash
st.set_page_config(page_title="RAG con Ollama (Ligero)", layout="centered")
st.title("📄🔗 App de Preguntas y Respuestas RAG con Ollama (phi3)")
st.markdown(
    "Sube un archivo PDF o TXT. Haz preguntas. Las respuestas son generadas usando el modelo ligero [phi3](https://ollama.com/library/phi3) a través de Ollama."
)
```
#  2. Verificación del estado de Ollama 
```bash
def is_ollama_running():
    """
    Verifica si el servidor de Ollama está activo y en ejecución en localhost:11434.
    Si no está en ejecución, muestra un error y detiene la aplicación.
    """
    try:
        r = requests.get("http://localhost:11434")
        return r.status_code == 200
    except Exception:
        return False

if not is_ollama_running():
    st.error(
        "¡Ollama no se está ejecutando! Por favor, abre una terminal y ejecuta:\n\n"
        "`ollama run phi3`\n\nLuego, reinicia esta aplicación."
    )
    st.stop()  # Detiene la app si Ollama no está funcionando
```

# 3. Entradas de usuario: Carga de archivo y pregunta
```bash
uploaded_file = st.file_uploader("Sube tu archivo PDF o TXT", type=["pdf", "txt"])
query = st.text_input("Haz una pregunta sobre tu documento:")
```

#  4. Almacenar la VectorDB en el estado de la sesión de Streamlit 
```bash
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def process_file(uploaded_file):
    """
    Carga el documento subido, lo divide en fragmentos, los vectoriza usando phi3
    y construye una base de datos vectorial con Chroma.
    """
    suffix = "." + uploaded_file.name.split(".")[-1]
    # Guarda el archivo subido en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Elige el cargador según la extensión del archivo
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load_and_split()  # Divide el documento en pequeños trozos de texto

    # Usa el modelo phi3 de Ollama para vectorizar los trozos de texto
    embeddings = OllamaEmbeddings(model="phi3")  # Ligero y rápido

    # Crea un directorio temporal para la base de datos vectorial de Chroma
    chroma_dir = tempfile.mkdtemp()
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=chroma_dir)
    return vectordb, chroma_dir
```

# 5. Manejar la carga del documento 
```bash
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Procesando tu documento (vectorizando)..."):
        vectordb, chroma_dir = process_file(uploaded_file)
        st.session_state.vectorstore = vectordb
        st.session_state.chroma_dir = chroma_dir
    st.success("✅ ¡Documento procesado! Haz tus preguntas abajo.")
```

#  6. Pipeline principal de Preguntas y Respuestas (RAG) 
```bash
if query and st.session_state.vectorstore:
    with st.spinner("Generando respuesta con phi3..."):
        llm = Ollama(
            model="phi3",                     # Usa phi3, el LLM ligero
            base_url="http://localhost:11434",
            temperature=0.1,                  # Temperatura baja: respuestas más factuales
            max_tokens=400,                   # Longitud de respuesta razonable
        )
        # RetrievalQA de LangChain: combina la recuperación con la generación del LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",               # Método de recuperación simple
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True      # Muestra qué fragmentos del doc se usaron
        )
        try:
            result = qa_chain(query)
            st.subheader("💡 Respuesta")
            st.write(result["result"])
            # Muestra los fragmentos de texto recuperados como contexto
            with st.expander("🔎 Ver el contexto recuperado"):
                for i, doc in enumerate(result['source_documents']):
                    st.markdown(f"**Fragmento {i+1}:**\n\n{doc.page_content}")
        except Exception as e:
            st.error(f"Error durante la generación de la respuesta: {str(e)}")
```

# --- 7. (Opcional) Limpieza: Elimina archivos/DBs temporales si lo deseas ---

```bash
st.markdown("---")
st.markdown(
    "Ligero y local — todo permanece en tu ordenador. Creado con [Ollama](https://ollama.com) y [LangChain](https://python.langchain.com/)."
)
```

Solución de Problemas

¿Ollama no está en ejecución?
Abre una terminal y ejecuta: ollama run phi3
¿No hay respuesta o la respuesta es muy lenta?
Asegúrate de que Ollama haya terminado de descargar el modelo.
Intenta reiniciar la aplicación de Streamlit.
Para archivos PDF grandes, espera unos segundos adicionales para la vectorización.
Usar un modelo diferente:
Cambia "phi3" en el código por cualquier otro nombre de modelo que hayas descargado con Ollama (por ejemplo, "llama2", "mistral", etc.).
Notas y Personalización
Soporta archivos PDF y TXT.
Solo local: Todos los archivos y el procesamiento son locales, sin acceso a la nube ni fugas de datos.
La base de datos vectorial es temporal: Cada nueva carga crea un nuevo almacén de vectores.
Funcionalidades avanzadas: Puedes añadir autenticación, historial, soporte para múltiples archivos, modo de chat o renderizado de Markdown.


# Créditos

Ollama

LangChain

ChromaDB

Streamlit

