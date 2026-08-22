from langchain_groq import ChatGroq
from retrieval import recuperar_contexto

MODELO_LLM = "openai/gpt-oss-120b"

PROMPT_SISTEMA = """Você é um assistente que responde perguntas do usuário com base \
APENAS no contexto de documentos fornecido abaixo.

Regras obrigatórias:
1. Responda somente com informações presentes no contexto. Nunca use \
conhecimento externo, suposições ou invente informações.
2. Se o contexto não tiver informação suficiente, diga claramente que não \
encontrou a resposta nos documentos disponíveis, em vez de arriscar um chute.
3. Seja claro, direto e objetivo.
4. Não repita o contexto bruto na resposta, apenas use-o como base.

Contexto recuperado:
{contexto}
"""

MENSAGEM_SEM_CONTEXTO = "Não encontrei essa informação nos documentos disponíveis."

#Função que gera a resposta da IA RAG
def gerar_resposta(pergunta: str, historico: list = None) -> dict:

    contexto, fontes = recuperar_contexto(pergunta)

    #Verificação se a IA conseguiu o contexto ou não
    if not contexto:
        return {"resposta": MENSAGEM_SEM_CONTEXTO, "fontes": [], "contexto_encontrado": False}

    #Fazendo a pergunta junto com o contexto
    mensagens = [{"role": "system", "content": PROMPT_SISTEMA.format(contexto=contexto)}]

    if historico:
        for msg in historico:
            mensagens.append({"role": msg["role"], "content": msg["content"]})

    mensagens.append({"role": "user", "content": pergunta})

    llm = ChatGroq(model=MODELO_LLM, temperature=0)
    resposta_llm = llm.invoke(mensagens)

    #Validação para ver se não voltou vazia
    texto_resposta = _validar_resposta(resposta_llm.content)
    return {"resposta": texto_resposta, "fontes": fontes, "contexto_encontrado": True}


#Função que valida a resposta, se ela está vazia ou não
def _validar_resposta(texto_resposta: str) -> str:
    if not texto_resposta or not texto_resposta.strip():
        return MENSAGEM_SEM_CONTEXTO
    return texto_resposta.strip()


#Função para formatar a resposta a ssim gerar a resposta final (Rsumos + Fontes)
def formatar_resposta_final(resultado: dict) -> str:
    resposta = resultado["resposta"]
    fontes = resultado["fontes"]

    if not resultado["contexto_encontrado"] or not fontes:
        return resposta

    linhas_fontes = sorted({f"- {f['arquivo']} (página {f['pagina']})" for f in fontes})
    return f"{resposta}\n\nFontes consultadas:\n" + "\n".join(linhas_fontes)