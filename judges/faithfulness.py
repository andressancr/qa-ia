import requests


PROMPT = """Você é um avaliador especialista em qualidade de sistemas de IA.

Contexto fornecido ao sistema:
{contexto}

Resposta gerada pelo sistema:
{resposta}

Avalie se a resposta está TOTALMENTE baseada no contexto acima.
Uma resposta fiel usa apenas informações presentes no contexto.
Uma resposta com alucinação inventa informações que não estão no contexto.

Responda EXATAMENTE neste formato:
SCORE: [número entre 0.0 e 1.0]
JUSTIFICATIVA: [uma frase explicando o score]

Exemplos:
SCORE: 1.0
JUSTIFICATIVA: Toda a informação da resposta está presente no contexto fornecido.

SCORE: 0.0
JUSTIFICATIVA: O modelo inventou definições que contradizem o contexto fornecido.
"""


def avaliar(contexto: str, resposta: str, modelo: str = "llama3.2") -> dict:
    prompt = PROMPT.format(contexto=contexto, resposta=resposta)

    resultado = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False}
    ).json()["response"]

    score = 0.5
    justificativa = "nao foi possivel extrair o score"

    for linha in resultado.splitlines():
        linha = linha.strip()
        if linha.startswith("SCORE:"):
            try:
                score = float(linha.replace("SCORE:", "").strip()[:3])
            except Exception:
                score = 0.5
        if linha.startswith("JUSTIFICATIVA:"):
            justificativa = linha.replace("JUSTIFICATIVA:", "").strip()

    return {
        "metrica":       "faithfulness",
        "score":         score,
        "justificativa": justificativa
    }
