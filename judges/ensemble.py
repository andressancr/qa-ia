import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges import relevancy, faithfulness, toxicity, corretude


MODELOS = ["llama3.2", "mistral"]


def media(scores: list) -> float:
    return round(sum(scores) / len(scores), 2)


def avaliar_ensemble(
    pergunta: str,
    resposta: str,
    contexto: str,
    esperado: str
) -> dict:
    resultados = {modelo: {} for modelo in MODELOS}

    for modelo in MODELOS:
        resultados[modelo]["relevancy"]    = relevancy.avaliar(pergunta, resposta, modelo)
        resultados[modelo]["faithfulness"] = faithfulness.avaliar(contexto, resposta, modelo)
        resultados[modelo]["toxicity"]     = toxicity.avaliar(resposta, modelo)
        resultados[modelo]["corretude"]    = corretude.avaliar(resposta, esperado, modelo)

    ensemble = {}
    for metrica in ["relevancy", "faithfulness", "toxicity", "corretude"]:
        scores        = [resultados[m][metrica]["score"]         for m in MODELOS]
        justificativas = [f"[{m}] {resultados[m][metrica]['justificativa']}" for m in MODELOS]

        ensemble[metrica] = {
            "score":         media(scores),
            "scores_por_modelo": {m: resultados[m][metrica]["score"] for m in MODELOS},
            "justificativas": justificativas,
            "concordancia":  abs(scores[0] - scores[1]) <= 0.2
        }

    return {
        "pergunta":  pergunta,
        "resposta":  resposta[:150],
        "modelos":   MODELOS,
        "ensemble":  ensemble
    }


def imprimir_resultado(resultado: dict):
    print(f"\nPergunta: {resultado['pergunta']}")
    print(f"Resposta: {resultado['resposta'][:100]}...")
    print(f"Modelos:  {' + '.join(resultado['modelos'])}")
    print("-" * 60)

    for metrica, dados in resultado["ensemble"].items():
        scores_str = " | ".join(
            f"{m}: {s}" for m, s in dados["scores_por_modelo"].items()
        )
        concordancia = "concordam" if dados["concordancia"] else "DIVERGEM"
        print(f"  {metrica:<15} media: {dados['score']}  [{scores_str}]  [{concordancia}]")
        for j in dados["justificativas"]:
            print(f"    {j}")
    print("-" * 60)
