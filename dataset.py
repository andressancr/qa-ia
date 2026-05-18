from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
import requests

lf = get_client()

CONTEXTO = """
RAG significa Retrieval-Augmented Generation.
E uma tecnica de inteligencia artificial que combina busca em base de dados
com geracao de texto por um LLM.
O sistema primeiro busca documentos relevantes e depois usa esse contexto
para gerar uma resposta mais precisa e fundamentada.
Foi popularizado pelo paper da Meta AI em 2020.

LLM significa Large Language Model.
E um modelo de linguagem treinado em grandes volumes de texto.
Exemplos: GPT, Claude, Gemini, Llama.
Sozinho ele nao busca dados externos, apenas responde com o que aprendeu.

Agente de IA e um sistema onde o LLM decide que acoes tomar.
O agente usa ferramentas externas como APIs, bancos de dados e buscas.
Ele pode iterar em loops ate completar uma tarefa complexa.

Faithfulness mede se a resposta esta baseada no contexto fornecido.
Score alto significa que o modelo nao inventou informacoes.
Score baixo significa que o modelo alucionou.

Hallucination ou alucinacao e quando o modelo inventa informacoes
que nao existem no contexto ou que sao falsas.
"""

DATASET = [
    {
        "pergunta": "O que e RAG em inteligencia artificial?",
        "esperado": "Retrieval-Augmented Generation"
    },
    {
        "pergunta": "O que significa LLM?",
        "esperado": "Large Language Model"
    },
    {
        "pergunta": "O que e um agente de IA?",
        "esperado": "sistema onde o LLM decide acoes e usa ferramentas"
    },
    {
        "pergunta": "O que e faithfulness em QA de IA?",
        "esperado": "mede se a resposta esta baseada no contexto"
    },
    {
        "pergunta": "O que e alucinacao em IA?",
        "esperado": "quando o modelo inventa informacoes falsas"
    },
]


def chamar_ollama(prompt: str) -> str:
    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    return resposta.json()["response"]


def avaliar_relevancy(pergunta: str, resposta: str) -> float:
    prompt = (
        f"Pergunta: {pergunta}\n"
        f"Resposta: {resposta}\n\n"
        "A resposta resolve diretamente a pergunta? "
        "Retorne APENAS um numero entre 0.0 e 1.0. Nada mais."
    )
    resultado = chamar_ollama(prompt)
    try:
        return float(resultado.strip()[:3])
    except Exception:
        return 0.5


def avaliar_faithfulness(contexto: str, resposta: str) -> float:
    prompt = (
        f"Contexto: {contexto}\n"
        f"Resposta: {resposta}\n\n"
        "A resposta esta TOTALMENTE baseada no contexto fornecido acima? "
        "Retorne APENAS um numero entre 0.0 e 1.0. "
        "1.0 = completamente fiel. 0.0 = inventou informacoes. Nada mais."
    )
    resultado = chamar_ollama(prompt)
    try:
        return float(resultado.strip()[:3])
    except Exception:
        return 0.5


def avaliar_corretude(resposta: str, esperado: str) -> float:
    prompt = (
        f"Resposta do modelo: {resposta}\n"
        f"Resposta esperada: {esperado}\n\n"
        "A resposta do modelo contem a informacao esperada? "
        "Retorne APENAS um numero entre 0.0 e 1.0. Nada mais."
    )
    resultado = chamar_ollama(prompt)
    try:
        return float(resultado.strip()[:3])
    except Exception:
        return 0.5


@observe()
def testar(pergunta: str, esperado: str) -> dict:
    resposta = chamar_ollama(pergunta)

    score_relevancy    = avaliar_relevancy(pergunta, resposta)
    score_faithfulness = avaliar_faithfulness(CONTEXTO, resposta)
    score_corretude    = avaliar_corretude(resposta, esperado)

    lf.score_current_trace(name="relevancy",    value=score_relevancy)
    lf.score_current_trace(name="faithfulness", value=score_faithfulness)
    lf.score_current_trace(name="corretude",    value=score_corretude)

    return {
        "relevancy":    score_relevancy,
        "faithfulness": score_faithfulness,
        "corretude":    score_corretude,
        "resposta":     resposta[:150]
    }


# roda o dataset completo
print("=" * 60)
print("RODANDO DATASET DE QA")
print("=" * 60)

resultados = []
for item in DATASET:
    print(f"\nPergunta: {item['pergunta']}")
    resultado = testar(item["pergunta"], item["esperado"])
    resultados.append(resultado)
    print(f"Relevancy:    {resultado['relevancy']}")
    print(f"Faithfulness: {resultado['faithfulness']}")
    print(f"Corretude:    {resultado['corretude']}")
    print("-" * 40)

# resumo final
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
total = len(resultados)
avg_relevancy    = sum(r["relevancy"]    for r in resultados) / total
avg_faithfulness = sum(r["faithfulness"] for r in resultados) / total
avg_corretude    = sum(r["corretude"]    for r in resultados) / total

print(f"Total de testes:       {total}")
print(f"Relevancy media:       {avg_relevancy:.2f}")
print(f"Faithfulness media:    {avg_faithfulness:.2f}")
print(f"Corretude media:       {avg_corretude:.2f}")

passou = avg_faithfulness >= 0.7 and avg_relevancy >= 0.7
print(f"\nResultado: {'PASSOU' if passou else 'FALHOU'}")

lf.flush()
