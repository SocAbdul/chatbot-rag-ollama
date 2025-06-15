# --- Importaciones: Librerías principales para la app ---
import streamlit as st                  # Para la interfaz web
import os
import tempfile                         # Para el manejo de archivos temporales
import requests                         # Para verificar si el servidor de Ollama está en ejecución

# --- LangChain y dependencias para el pipeline RAG ---
from langchain.document_loaders import PyPDFLoader, TextLoader    # Para cargar archivos PDF/TXT
from langchain.embeddings import OllamaEmbeddings                 # Para vectorizar texto usando modelos de Ollama
from langchain.vectorstores import Chroma                         # Para almacenar y buscar los vectores (embeddings)
from langchain.llms import Ollama                                 # Para conectarse al LLM de Ollama
from langchain.chains import RetrievalQA                          # Cadena de LangChain para RAG

# --- 1. Configuración de la página de Streamlit ---
st.set_page_config(page_title="RAG con Ollama (Ligero)", layout="centered")
st.title("📄🔗 App de Preguntas y Respuestas RAG con Ollama (phi3)")
st.markdown(
    "Sube un archivo PDF o TXT. Haz preguntas. Las respuestas son generadas usando el modelo ligero [phi3](https://ollama.com/library/phi3) a través de Ollama."
)

# --- 2. Verificación del estado de Ollama ---
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

# --- 3. Entradas de usuario: Carga de archivo y pregunta ---
uploaded_file = st.file_uploader("Sube tu archivo PDF o TXT", type=["pdf", "txt"])
query = st.text_input("Haz una pregunta sobre tu documento:")

# --- 4. Almacenar la Base de Datos Vectorial en el estado de la sesión ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def process_file(uploaded_file):
    """
    Carga el documento subido, lo divide en fragmentos, los vectoriza usando phi3
    y construye una base de datos vectorial con Chroma.
    """
    # Extrae la extensión del archivo para usarla en el archivo temporal
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
    embeddings = OllamaEmbeddings(model="phi3")  # Modelo ligero y rápido

    # Crea un directorio temporal para la base de datos vectorial de Chroma
    chroma_dir = tempfile.mkdtemp()
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=chroma_dir)
    return vectordb, chroma_dir

# --- 5. Manejar la carga del documento ---
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Procesando tu documento (vectorizando)..."):
        vectordb, chroma_dir = process_file(uploaded_file)
        st.session_state.vectorstore = vectordb
        st.session_state.chroma_dir = chroma_dir
    st.success("✅ ¡Documento procesado! Haz tus preguntas abajo.")

# --- 6. Pipeline principal de Preguntas y Respuestas RAG ---
if query and st.session_state.vectorstore:
    with st.spinner("Generando respuesta con phi3..."):
        # Inicializa el LLM de Ollama
        llm = Ollama(
            model="phi3",                     # Usa phi3, el LLM ligero
            base_url="http://localhost:11434",
            temperature=0.1,                  # Temperatura baja para respuestas más factuales
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

# --- 7. Limpieza: Eliminar archivos temporales al salir de la app ---
def cleanup():
    """
    Limpia los archivos y directorios temporales creados durante la ejecución de la app.
    """
    if "chroma_dir" in st.session_state:
        chroma_dir = st.session_state.chroma_dir
        if os.path.exists(chroma_dir):
            # Elimina de forma segura el directorio temporal y su contenido
            import shutil
            shutil.rmtree(chroma_dir)

# Registra la función de limpieza para que se ejecute al salir de la app
import atexit
atexit.register(cleanup)

# --- 8. Pie de página ---
st.markdown("---")
st.markdown(
    "Ligero y local — todo permanece en tu ordenador. Creado con [Ollama](https://ollama.com) y [LangChain](https://python.langchain.com/)."
)
