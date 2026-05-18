import requests


PROMPT = """Você é um avaliador especialista em qualidade de sistemas de IA.

Resposta gerada pelo sistema:
{resposta}

Resposta esperada (gabarito):
{esperado}

Avalie se a resposta do sistema contém a informação esperada.
Não precisa ser idêntica — avalie se o conteúdo essencial está presente.

Responda EXATAMENTE neste formato:
SCORE: [número entre 0.0 e 1.0]
JUSTIFICATIVA: [uma frase explicando o score]

Exemplos:
SCORE: 1.0
JUSTIFICATIVA: A resposta contém exatamente a informação esperada sobre o tema.

SCORE: 0.1
JUSTIFICATIVA: A resposta não menciona a informação esperada e apresenta dados incorretos.
"""


def avaliar(resposta: str, esperado: str, modelo: str = "llama3.2") -> dict:
    prompt = PROMPT.format(resposta=resposta, esperado=esperado)

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
        "metrica":       "corretude",
        "score":         score,
        "justificativa": justificativa
    }
