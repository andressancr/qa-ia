import requests


PROMPT = """Você é um avaliador especialista em qualidade de sistemas de IA.

Pergunta do usuário: {pergunta}
Resposta do sistema: {resposta}

Avalie se a resposta resolve diretamente a pergunta do usuário.

Responda EXATAMENTE neste formato:
SCORE: [número entre 0.0 e 1.0]
JUSTIFICATIVA: [uma frase explicando o score]

Exemplos:
SCORE: 0.9
JUSTIFICATIVA: A resposta aborda diretamente o tema perguntado com detalhes relevantes.

SCORE: 0.2
JUSTIFICATIVA: A resposta foge do tema e não responde o que foi perguntado.
"""


def avaliar(pergunta: str, resposta: str, modelo: str = "llama3.2") -> dict:
    prompt = PROMPT.format(pergunta=pergunta, resposta=resposta)

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
        "metrica":       "relevancy",
        "score":         score,
        "justificativa": justificativa
    }
