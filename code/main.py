from dotenv import load_dotenv
load_dotenv()

from retrieval import recuperar_contexto
from vectorstore import criar_db, db_existe

if __name__ == "__main__":
    if not db_existe():
        print("Base vetorial não encontrada, criando a partir de /docs...")
        criar_db()
    else:
        print("Base vetorial já existe, pulando ingestão.")

    print("\nAgente pronto. Digite sua pergunta (ou 'sair' para encerrar).\n")

    pergunta = input("Digite sua pergunta: ")
    contexto, fontes = recuperar_contexto(pergunta)
    print(contexto)
    print(fontes)