from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import os

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"
embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_DB = os.path.join(BASE_DIR, "..", "db")
CAMINHO_DOCS = os.path.join(BASE_DIR, "..", "docs")

#Função para criar o banco vetorial já com o texto separado por chunks semânticas dentro sele
def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks2(documentos)
    vetorizar_chunks(chunks)

#Função para carregar os documentos (Somente PDF)
def carregar_documentos():
    print("Caminho absoluto:", os.path.abspath(CAMINHO_DOCS))
    print("Existe?", os.path.isdir(CAMINHO_DOCS))
    print("Arquivos na pasta:", os.listdir(CAMINHO_DOCS) if os.path.isdir(CAMINHO_DOCS) else "N/A")

    carregador = PyPDFDirectoryLoader(CAMINHO_DOCS, glob="*.pdf")
    documentos = carregador.load()
    return documentos

#Função para dividir os documentos em chunks
def dividir_chunks2(documentos):
    embeddings_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Iniciando split semântico...")
    separador_documentos = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=embeddings_tokenizer,
        chunk_size=embeddings_tokenizer.max_len_single_sentence,
        chunk_overlap=0,
    )
    print("Splitter criado, processando documentos...")
    chunks = separador_documentos.split_documents(documentos)
    print("Split concluído!")
    if chunks:
        print(chunks[13] if len(chunks) > 13 else chunks[0])
    print(len(chunks))
    return chunks

#Função para subir tudo no banco vetorial do Chroma e logo abaixo funções para validações
def vetorizar_chunks(chunks):
    Chroma.from_documents(chunks, embed_model, persist_directory=CAMINHO_DB)

