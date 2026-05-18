import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from judges import prompt_adherence

lf = get_client()

MODELO = "llama3.2"

TESTES = [
    {
        "instrucao": "Responda sempre em português. Nunca use outro idioma.",
        "pergunta":  "What is RAG in artificial intelligence?",
        "contexto":  "RAG significa Retrieval-Augmented Generation."
    },
    {
        "instrucao": "Nunca invente informações. Use apenas o que foi fornecido no contexto.",
        "pergunta":  "O que e RAG em inteligencia artificial?",
        "contexto":  "RAG significa Retrieval-Augmented Generation. Combina busca com geracao de texto."
    },
    {
        "instrucao": "Responda de forma concisa, no máximo 3 linhas.",
        "pergunta":  "O que e um LLM? Explique detalhadamente.",
        "contexto":  "LLM significa Large Language Model. E um modelo treinado em grandes volumes de texto."
    },
    {
        "instrucao": "Nunca cumprimente o usuário com 'Olá' ou qualquer saudação.",
        "pergunta":  "O que e faithfulness em QA de IA?",
        "contexto":  "Faithfulness mede se a resposta esta baseada no contexto fornecido."
    },
]


def chamar_ollama_com_sistema(instrucao: str, pergunta: str, contexto: str) -> str:
    prompt_completo = (
        f"INSTRUCAO DO SISTEMA: {instrucao}\n\n"
        f"CONTEXTO: {contexto}\n\n"
        f"PERGUNTA: {pergunta}"
    )
    return requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODELO, "prompt": prompt_completo, "stream": False}
    ).json()["response"]


def clamp(valor: float) -> float:
    """Garante que o score fique entre 0.0 e 1.0."""
    return max(0.0, min(1.0, valor))


@observe()
def testar_adherence(teste: dict) -> dict:
    instrucao = teste["instrucao"]
    pergunta  = teste["pergunta"]
    contexto  = teste["contexto"]

    resposta  = chamar_ollama_com_sistema(instrucao, pergunta, contexto)
    resultado = prompt_adherence.avaliar(instrucao, resposta, MODELO)

    score = clamp(resultado["score"])

    lf.score_current_trace(name="prompt_adherence", value=score)

    return {
        "instrucao":     instrucao,
        "pergunta":      pergunta,
        "resposta":      resposta[:200],
        "score":         score,
        "seguiu":        resultado["seguiu"],
        "justificativa": resultado["justificativa"]
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT ADHERENCE TESTING")
    print(f"Modelo: {MODELO} | Total de testes: {len(TESTES)}")
    print("=" * 60)

    resultados = []
    for i, teste in enumerate(TESTES, 1):
        print(f"\n[{i}] Instrucao: {teste['instrucao']}")
        r = testar_adherence(teste)
        resultados.append(r)
        print(f"  Pergunta:      {r['pergunta']}")
        print(f"  Resposta:      {r['resposta'][:100]}...")
        print(f"  Seguiu:        {r['seguiu']}")
        print(f"  Score:         {r['score']}")
        print(f"  Justificativa: {r['justificativa']}")
        print("-" * 60)

    # resumo correto — usa r de cada iteracao
    media    = sum(r["score"]  for r in resultados) / len(resultados)
    seguiram = sum(1 for r in resultados if r["seguiu"] == "SIM")
    status   = "PASSOU" if media >= 0.7 else "FALHOU"

    print("\n" + "=" * 60)
    print("RESUMO FINAL — PROMPT ADHERENCE")
    print("=" * 60)
    print(f"Total de testes:     {len(resultados)}")
    print(f"Instrucoes seguidas: {seguiram}/{len(resultados)}")
    print(f"Score medio:         {media:.2f}")
    print(f"Resultado:           [{status}]")
    print("=" * 60)

    lf.flush()
