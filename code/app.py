import gradio as gr
from dotenv import load_dotenv
load_dotenv()
from vectorstore import db_existe, criar_db
from llm import gerar_resposta, formatar_resposta_final

if not db_existe():
    print("Base vetorial não encontrada, criando a partir de /docs...")
    criar_db()


def responder(pergunta, historico):
    resultado = gerar_resposta(pergunta, historico)
    return formatar_resposta_final(resultado)


demo = gr.ChatInterface(
    fn=responder,
    title="🤖 Assistente Interno",
    description="Você está conversando com um agente de IA. As respostas são baseadas nos documentos internos da empresa.",
)

if __name__ == "__main__":
    demo.launch()