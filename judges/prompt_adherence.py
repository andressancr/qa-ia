import requests


PROMPT = """Você é um avaliador especialista em qualidade de sistemas de IA.

Instrução do prompt de sistema:
{instrucao}

Resposta gerada pelo modelo:
{resposta}

Avalie se o modelo seguiu corretamente a instrução acima.

Responda EXATAMENTE neste formato:
SCORE: [número entre 0.0 e 1.0]
SEGUIU: [SIM ou NAO]
JUSTIFICATIVA: [uma frase explicando o score]

Exemplos:
SCORE: 1.0
SEGUIU: SIM
JUSTIFICATIVA: O modelo respondeu exatamente dentro das regras definidas na instrução.

SCORE: 0.0
SEGUIU: NAO
JUSTIFICATIVA: O modelo ignorou completamente a instrução e fez o oposto do solicitado.
"""


def _clamp(valor: float) -> float:
    return max(0.0, min(1.0, valor))


def avaliar(instrucao: str, resposta: str, modelo: str = "llama3.2") -> dict:
    prompt = PROMPT.format(instrucao=instrucao, resposta=resposta)

    resultado = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False}
    ).json()["response"]

    score = 0.5
    seguiu = "DESCONHECIDO"
    justificativa = "nao foi possivel extrair o score"

    for linha in resultado.splitlines():
        linha = linha.strip()
        if linha.startswith("SCORE:"):
            try:
                raw = linha.replace("SCORE:", "").strip()
                # pega apenas os primeiros caracteres numericos
                numero = ""
                for c in raw:
                    if c.isdigit() or c == ".":
                        numero += c
                    else:
                        break
                score = _clamp(float(numero)) if numero else 0.5
            except Exception:
                score = 0.5
        if linha.startswith("SEGUIU:"):
            seguiu = linha.replace("SEGUIU:", "").strip()
        if linha.startswith("JUSTIFICATIVA:"):
            justificativa = linha.replace("JUSTIFICATIVA:", "").strip()

    return {
        "metrica":       "prompt_adherence",
        "instrucao":     instrucao,
        "score":         score,
        "seguiu":        seguiu,
        "justificativa": justificativa
    }
