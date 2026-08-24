# 🤖 Assistente Interno com RAG

Agente de IA conversacional que responde perguntas com base **exclusivamente** nos documentos internos da empresa (PDFs), usando a técnica de **RAG (Retrieval-Augmented Generation)** com reranking semântico para aumentar a precisão das respostas.

O agente foi construído para reduzir alucinações: ele nunca deve responder com base em conhecimento externo do modelo — apenas com o que está de fato presente nos documentos indexados. Quando não encontra a informação, ele avisa isso claramente ao usuário.

---

## 🏗️ Arquitetura da solução

O projeto segue um pipeline clássico de RAG, dividido em módulos:

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

### Fluxo da consulta (Retrieval)

1. **Busca por similaridade**: `buscar_candidatos()` recupera os `k=20` chunks mais similares à pergunta no banco vetorial Chroma.
2. **Reranking**: `rerankear()` usa um Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) para reordenar esses candidatos por relevância real em relação à pergunta, mantendo apenas os `top_n=5` melhores.
3. **Montagem do contexto**: `montar_contexto()` filtra os trechos abaixo do limiar de confiança e monta o texto final, junto com a lista de fontes (arquivo + página).
4. **Geração da resposta**: `gerar_resposta()` monta o prompt do sistema (com o contexto injetado) e envia ao LLM via Groq, incluindo o histórico da conversa.
5. **Formatação final**: `formatar_resposta_final()` adiciona a lista de fontes consultadas ao final da resposta.

Se nenhum contexto relevante for encontrado, o agente responde com uma mensagem padrão informando que não encontrou a informação nos documentos — sem tentar "chutar" uma resposta.

---

## 🛠️ Tecnologias e ferramentas utilizadas

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python |
| Orquestração RAG | [LangChain](https://www.langchain.com/) (`langchain-community`, `langchain-text-splitters`) |
| LLM | Groq API (`langchain-groq`), modelo `openai/gpt-oss-120b` |
| Embeddings | HuggingFace — `ibm-granite/granite-embedding-30m-english` (`langchain-huggingface`) |
| Banco vetorial | [Chroma](https://www.trychroma.com/) (`langchain-chroma`, `chromadb`) |
| Reranking | Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (`sentence-transformers`) |
| Leitura de PDFs | `PyPDFDirectoryLoader` (`pypdf`) |
| Interface web | [Gradio](https://www.gradio.dev/) (`gr.ChatInterface`) |
| Variáveis de ambiente | `python-dotenv` |

---

## ▶️ Instruções para executar o projeto

### 1. Pré-requisitos
- Python 3.10+
- Uma chave de API da [Groq](https://console.groq.com/keys)

### 2. Estrutura de pastas do projeto

Os arquivos Python ficam dentro da pasta `code/`. As pastas `db/` e `docs/` ficam **fora** de `code/`, no mesmo nível — é assim que `vectorstore.py` resolve os caminhos (`CAMINHO_DB`/`CAMINHO_DOCS` = pasta de `vectorstore.py` + `..` + `db`/`docs`).

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
├── docs/                # coloque aqui os PDFs a serem indexados
├── db/                  # criado automaticamente na primeira execução
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── oci/                 # scripts e guia de deploy na OCI
```

### 3. Instalação

```bash
cd RAG_Alura/code

# criar e ativar um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# instalar as dependências
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Crie um arquivo `.env` **dentro da pasta `code/`** com sua chave da Groq:

```
GROQ_API_KEY=sua_chave_aqui
```

> ⚠️ **Nunca** commite o arquivo `.env` no repositório. Ele já está no `.gitignore`.

### 5. Adicionar os documentos

Coloque os arquivos PDF que servirão de base de conhecimento dentro da pasta `docs/` (na raiz do projeto, fora de `code/`).

### 6. Executar

Rode os comandos a partir de dentro de `code/`:

**Via interface web (Gradio):**
```bash
cd code
python app.py
```
Isso abrirá uma interface de chat no navegador. Na primeira execução, o banco vetorial será criado automaticamente a partir dos PDFs em `../docs` (pode levar alguns minutos, dependendo da quantidade de documentos).

**Via terminal (CLI):**
```bash
cd code
python main.py
```
Digite suas perguntas diretamente no terminal. Digite `sair`, `exit` ou `quit` para encerrar.

---

## 💬 Exemplos de perguntas que o agente consegue responder

O agente responde a perguntas cujas respostas estejam **presentes nos documentos indexados** em `docs/`. Exemplos típicos para um assistente interno de empresa:

- "Qual é a política de reembolso de despesas de viagem?"
- "Quantos dias de férias eu tenho direito por ano?"
- "Como funciona o processo de onboarding de novos funcionários?"
- "Qual é o procedimento para solicitar home office?"
- "Quais são os benefícios oferecidos pela empresa?"

> 💡 As perguntas reais que o agente conseguirá responder dependem inteiramente do conteúdo dos PDFs colocados na pasta `docs/`.

---

## 📄 Exemplos de respostas geradas pelo agente

**Pergunta:** "Quantos dias de férias eu tenho direito por ano?"

**Resposta:**
```
De acordo com os documentos internos, o colaborador tem direito a 30 dias 
corridos de férias por ano, podendo ser divididos em até três períodos, 
sendo que um deles não pode ser inferior a 14 dias corridos.

Fontes consultadas:
- politica_ferias.pdf (página 2)
```

**Pergunta:** "Qual o horário de funcionamento do escritório aos sábados?"

**Resposta (quando a informação não está nos documentos):**
```
Não encontrei essa informação nos documentos disponíveis.
```

Esse comportamento é intencional: o agente é instruído a nunca inventar informações que não estejam no contexto recuperado, evitando respostas incorretas ou enganosas.

---

## ☁️ Deploy na nuvem (OCI)

O projeto inclui um `Dockerfile` e scripts prontos para publicar a aplicação na Oracle Cloud Infrastructure (Container Registry + Container Instances, ou Compute VM). Veja o passo a passo completo em [`oci/README-DEPLOY.md`](oci/README-DEPLOY.md).
