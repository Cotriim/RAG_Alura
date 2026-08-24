# Assistente Interno com RAG

Agente de IA conversacional que responde perguntas com base exclusivamente nos documentos internos de uma empresa (PDFs), usando RAG (Retrieval-Augmented Generation) com reranking semântico.

A regra é simples: o agente nunca responde com conhecimento próprio do modelo, só com o que está de fato nos documentos indexados. Se não encontra a informação, ele diz isso claramente em vez de tentar adivinhar.

---

## Arquitetura

Pipeline dividido em módulos:

```
┌─────────────────────┐
│   /docs (PDFs)       │
└──────────┬───────────┘
           │  ingestão (1x)
           ▼
┌─────────────────────┐
│   vectorstore.py     │
│  - Carrega PDFs      │
│  - Split em chunks   │
│  - Gera embeddings   │
│  - Persiste no Chroma│
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│      /db (Chroma)    │
└──────────┬───────────┘
           │  busca (a cada pergunta)
           ▼
┌─────────────────────┐
│    retrieval.py      │
│  - Busca top-k       │
│    (similaridade)    │
│  - Reranking com     │
│    Cross-Encoder     │
│  - Monta contexto    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│       llm.py         │
│  - Prompt com regras │
│  - Chama o LLM (Groq)│
│  - Valida resposta   │
│  - Formata fontes    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  app.py / main.py    │
│  - Interface Gradio  │
│    ou terminal (CLI) │
└─────────────────────┘
```

### Fluxo da consulta

1. `buscar_candidatos()` recupera os `k=20` chunks mais similares à pergunta no Chroma.
2. `rerankear()` usa um Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) pra reordenar por relevância real, mantendo os `top_n=5` melhores.
3. `montar_contexto()` filtra o que está abaixo do limiar de confiança e monta o texto final, junto com a lista de fontes (arquivo + página).
4. `gerar_resposta()` monta o prompt do sistema com o contexto injetado e chama o LLM via Groq, incluindo o histórico da conversa.
5. `formatar_resposta_final()` adiciona as fontes consultadas ao final da resposta.

Sem contexto relevante, o agente cai na resposta padrão avisando que não encontrou a informação — sem chutar.

---

## Stack

| Categoria             | Tecnologia                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------- |
| Linguagem             | Python                                                                                       |
| Orquestração RAG      | [LangChain](https://www.langchain.com/) (`langchain-community`, `langchain-text-splitters`) |
| LLM                   | Groq API (`langchain-groq`), modelo `openai/gpt-oss-120b`                                   |
| Embeddings            | HuggingFace — `ibm-granite/granite-embedding-30m-english` (`langchain-huggingface`)         |
| Banco vetorial        | [Chroma](https://www.trychroma.com/) (`langchain-chroma`, `chromadb`)                       |
| Reranking             | Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (`sentence-transformers`)              |
| Leitura de PDFs       | `PyPDFDirectoryLoader` (`pypdf`)                                                             |
| Interface web         | [Gradio](https://www.gradio.dev/) (`gr.ChatInterface`)                                      |
| Variáveis de ambiente | `python-dotenv`                                                                              |

---

## Rodando o projeto

### Pré-requisitos

- Python 3.10+
- Chave de API da [Groq](https://console.groq.com/keys)

### Estrutura de pastas

Os arquivos Python ficam em `code/`. `db/` e `docs/` ficam fora de `code/`, no mesmo nível — é assim que `vectorstore.py` resolve os caminhos (`CAMINHO_DB`/`CAMINHO_DOCS` = pasta de `vectorstore.py` + `..` + `db`/`docs`).

```
RAG_Alura/
├── code/
│   ├── .env             # sua chave da Groq (não versionar)
│   ├── app.py
│   ├── main.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── vectorstore.py
│   └── requirements.txt
├── docs/                # PDFs a serem indexados
├── db/                  # criado automaticamente na primeira execução
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── oci/                 # scripts e guia de deploy na OCI
```

### Instalação

```bash
cd RAG_Alura/code

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um `.env` dentro de `code/`:

```
GROQ_API_KEY=sua_chave_aqui
```

O `.env` já está no `.gitignore` — não commite essa chave.

### Documentos

Coloque os PDFs que servirão de base de conhecimento em `docs/`, na raiz do projeto (fora de `code/`).

### Executando

Via interface web (Gradio):

```bash
cd code
python app.py
```

Na primeira execução o banco vetorial é criado a partir dos PDFs em `../docs` — pode levar alguns minutos dependendo do volume de documentos.

Via terminal:

```bash
cd code
python main.py
```

`sair`, `exit` ou `quit` encerram a sessão.

---

## Exemplos de perguntas

Depende do conteúdo dos PDFs em `docs/`, mas para um assistente interno típico:

- "Qual é a política de reembolso de despesas de viagem?"
- "Quantos dias de férias eu tenho direito por ano?"
- "Como funciona o processo de onboarding de novos funcionários?"
- "Qual é o procedimento para solicitar home office?"
- "Quais são os benefícios oferecidos pela empresa?"

## Exemplo de resposta

**Pergunta:** "Quantos dias de férias eu tenho direito por ano?"

```
De acordo com os documentos internos, o colaborador tem direito a 30 dias
corridos de férias por ano, podendo ser divididos em até três períodos,
sendo que um deles não pode ser inferior a 14 dias corridos.

Fontes consultadas:
- politica_ferias.pdf (página 2)
```

Quando a informação não está nos documentos, a resposta é direta:

```
Não encontrei essa informação nos documentos disponíveis.
```

Esse comportamento é proposital — o agente é instruído a nunca preencher lacunas com suposições.

---

## Deploy (OCI)

O projeto inclui `Dockerfile` e scripts para publicar na Oracle Cloud Infrastructure (Container Registry + Container Instances, ou Compute VM). Passo a passo em [`oci/README-DEPLOY.md`](https://github.com/Cotriim/RAG_Alura/blob/main/oci/README-DEPLOY.md).
