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
"""


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


@observe()
def testar(pergunta: str) -> str:
    resposta = chamar_ollama(pergunta)

    score_relevancy    = avaliar_relevancy(pergunta, resposta)
    score_faithfulness = avaliar_faithfulness(CONTEXTO, resposta)

    # metodos corretos da v4
    lf.score_current_trace(name="relevancy",    value=score_relevancy)
    lf.score_current_trace(name="faithfulness", value=score_faithfulness)

    print(f"\nPergunta:     {pergunta}")
    print(f"Resposta:     {resposta[:200]}...")
    print(f"Relevancy:    {score_relevancy}")
    print(f"Faithfulness: {score_faithfulness}")
    print("-" * 50)
    return resposta


testar("O que e RAG em inteligencia artificial?")
lf.flush()
