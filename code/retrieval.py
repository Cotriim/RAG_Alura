from sentence_transformers import CrossEncoder
from vectorstore import carregar_db

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None  # carregado sob demanda


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

#Função para buscar possiveis chunks da vectorstore
def buscar_candidatos(pergunta: str, k: int = 20, filtro_metadados: dict = None):
    db = carregar_db()
    if filtro_metadados:
        return db.similarity_search_with_relevance_scores(pergunta, k=k, filter=filtro_metadados)
    return db.similarity_search_with_relevance_scores(pergunta, k=k)

#Função para afunilar as possiveis chunks da vectorstore, assim deixando o contexto mais preciso
def rerankear(pergunta: str, candidatos: list, top_n: int = 5):
    if not candidatos:
        return []
    reranker = _get_reranker()
    pares = [(pergunta, doc.page_content) for doc, _ in candidatos]
    scores = reranker.predict(pares)
    candidatos_com_score = list(zip([doc for doc, _ in candidatos], scores))
    candidatos_com_score.sort(key=lambda item: item[1], reverse=True)
    return candidatos_com_score[:top_n]

#Função para montar o contexto...
def montar_contexto(candidatos_rerankeados: list, limiar_confianca: float = 0.0):
    trechos_validos = [(doc, score) for doc, score in candidatos_rerankeados if score >= limiar_confianca]
    if not trechos_validos:
        return None, []

    partes, fontes = [], []
    for doc, _ in trechos_validos:
        fonte = {
            "arquivo": doc.metadata.get("source", "desconhecido"),
            "pagina": doc.metadata.get("page", "N/A"),
        }
        fontes.append(fonte)
        partes.append(f"[Fonte: {fonte['arquivo']} - página {fonte['pagina']}]\n{doc.page_content}")

    return "\n\n---\n\n".join(partes), fontes


def recuperar_contexto(
    pergunta: str,
    k: int = 20,
    top_n: int = 5,
    filtro_metadados: dict = None,
    limiar_confianca: float = 0.0,
):
    candidatos = buscar_candidatos(pergunta, k=k, filtro_metadados=filtro_metadados)
    if not candidatos:
        return None, []

    rerankeados = rerankear(pergunta, candidatos, top_n=top_n)
    return montar_contexto(rerankeados, limiar_confianca=limiar_confianca)