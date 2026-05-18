# QA de Inteligência Artificial com Langfuse e Ollama

Pipeline completo de Quality Assurance para sistemas de IA, com avaliação automática de respostas de LLMs usando métricas de qualidade e observabilidade via Langfuse.

---

## Background

Este projeto foi desenvolvido com base na experiência prática em QA de LLMs adquirida durante atuação no projeto **LaMDA (Google)**, onde as mesmas métricas de avaliação — Faithfulness, Relevancy, Groundedness e Toxicity — eram aplicadas manualmente no processo de treinamento e alinhamento de grandes modelos de linguagem.

### O que era feito manualmente no Google, hoje automatizado aqui

| No projeto LaMDA (Google) | Neste projeto (automatizado) |
|---|---|
| Atribuição manual de scores de **Sensibleness** — coerência lógica do diálogo | `relevancy` — avalia se a resposta resolve a pergunta |
| Atribuição manual de scores de **Groundedness** — factualidade ancorada em fontes | `faithfulness` — detecta se o modelo alucionou |
| Classificação de **Safety & Toxicity** — conteúdo ofensivo e desinformação | `toxicity` — monitora respostas inadequadas |
| Avaliação de **Specificity** — penalização de respostas genéricas | `corretude` — verifica se a resposta esperada foi atingida |
| Coleta e rotulagem de dados para **RLHF e Fine-Tuning** | dataset com `expected_output` para experimentos no Langfuse |
| Análise de **Taxonomia e Intenção** — validação de nuances e contexto | fluxo de QA cobrindo entrada, prompt, RAG, LLM, tools e resposta final |

> No Google eu fazia manualmente o que hoje ferramentas como Langfuse, DeepEval e RAGAS automatizam. Avaliava respostas de LLMs atribuindo scores em critérios como coerência, relevância, factualidade e toxicidade — exatamente as métricas que o mercado hoje chama de Faithfulness, Relevancy, Groundedness e Toxicity. O LaMDA foi a base do que se tornou o Bard e depois o Gemini.

---

## O que é este projeto

Implementação de um framework de QA de IA do zero, sem depender de APIs pagas. Utiliza modelos de linguagem locais via Ollama e rastreia todas as execuções no Langfuse, permitindo avaliar automaticamente a qualidade das respostas geradas por LLMs.

O objetivo é detectar problemas como alucinação, respostas irrelevantes e baixa fidelidade ao contexto — exatamente como é feito em projetos reais de IA em produção.

---

## Tecnologias utilizadas

| Tecnologia | Função |
|---|---|
| [Ollama](https://ollama.com) | Roda modelos de LLM localmente (sem custo) |
| [Langfuse](https://langfuse.com) | Observabilidade, rastreamento de traces e scores |
| Python 3.14 | Linguagem principal |
| llama3.2 | Modelo de linguagem usado nos testes |

---

## Estrutura do projeto

```
qa-ia/
  teste.py      # teste simples de uma pergunta com scores de qualidade
  dataset.py    # dataset com 5 perguntas e avaliação completa
  .gitignore    # protege o .env com as chaves de API
  .env          # chaves de API (não sobe pro GitHub)
```

---

## Métricas de avaliação implementadas

### Relevancy
Avalia se a resposta resolve diretamente a pergunta do usuário.
- Score 0.0 a 1.0
- Threshold mínimo: 0.7

### Faithfulness
Avalia se a resposta está totalmente baseada no contexto fornecido — detecta alucinação.
- Score 0.0 = modelo inventou informações
- Score 1.0 = completamente fiel ao contexto
- Threshold mínimo: 0.7

### Corretude
Avalia se a resposta contém a informação esperada (expected output).
- Score 0.0 a 1.0
- Threshold mínimo: 0.7

---

## Como funciona

```
Pergunta do usuário
       |
   Ollama (llama3.2) gera resposta
       |
   LLM-as-Judge avalia:
       ├── Relevancy    → a resposta resolve a pergunta?
       ├── Faithfulness → está ancorada no contexto?
       └── Corretude    → contém a resposta esperada?
       |
   Scores enviados ao Langfuse
       |
   Dashboard com traces, scores e histórico completo
```

---

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e rodando
- Conta gratuita no [Langfuse](https://langfuse.com)

---

## Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/andressancr/qa-ia.git
cd qa-ia
```

### 2. Instale as dependências

```bash
pip install langfuse python-dotenv requests
```

### 3. Baixe o modelo

```bash
ollama pull llama3.2
```

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

As chaves são geradas em: **langfuse.com → Settings → API Keys**

### 5. Rode o teste simples

```bash
python teste.py
```

### 6. Rode o dataset completo

```bash
python dataset.py
```

---

## Exemplo de resultado

```
============================================================
RODANDO DATASET DE QA
============================================================

Pergunta: O que e RAG em inteligencia artificial?
Relevancy:    0.8
Faithfulness: 0.0
Corretude:    0.9

Pergunta: O que significa LLM?
Relevancy:    0.8
Faithfulness: 0.0
Corretude:    0.9

============================================================
RESUMO FINAL
============================================================
Total de testes:       5
Relevancy media:       0.80
Faithfulness media:    0.36
Corretude media:       0.84

Resultado: FALHOU
```

O resultado **FALHOU** indica que o modelo llama3.2 alucionou em várias respostas — inventou significados incorretos para RAG, LLM e outros termos que estavam claramente definidos no contexto. O faithfulness baixo (0.36) detectou exatamente esse comportamento — o mesmo tipo de problema identificado manualmente durante a atuação no projeto LaMDA.

---

## O que vem a seguir

- [ ] Adicionar métrica de Toxicity
- [ ] Implementar Context Precision e Context Recall
- [ ] Criar pipeline CI/CD com gate de qualidade (pytest + GitHub Actions)
- [ ] Testar com outros modelos (mistral, gemma, phi)
- [ ] Comparar resultados entre modelos no Langfuse

---

## Autor

Desenvolvido como parte dos estudos em QA de Inteligência Artificial, com base na experiência prática em avaliação de LLMs adquirida no projeto LaMDA (Google).
