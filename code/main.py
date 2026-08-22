from dotenv import load_dotenv
load_dotenv()

from vectorstore import criar_db, db_existe
from llm import gerar_resposta, formatar_resposta_final

if __name__ == "__main__":
    if not db_existe():
        print("Base vetorial não encontrada, criando a partir de /docs...")
        criar_db()
    else:
        print("Base vetorial já existe, pulando ingestão.")

    print("\nAgente pronto. Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        resultado = gerar_resposta(pergunta)
        print("\nAgente:", formatar_resposta_final(resultado), "\n")