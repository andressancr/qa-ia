import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from judges.ensemble import avaliar_ensemble, imprimir_resultado

lf = get_client()

THRESHOLDS = {
    "relevancy":    0.70,
    "faithfulness": 0.70,
    "corretude":    0.70,
    "toxicity":     0.30,
}


def carregar_dataset() -> list:
    caminho = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "golden_qa.json"
    )
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


@observe()
def avaliar_item_ensemble(item: dict) -> dict:
    resultado = avaliar_ensemble(
        pergunta=item["pergunta"],
        resposta="",
        contexto=item["contexto"],
        esperado=item["esperado"]
    )

    import requests
    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": item["pergunta"], "stream": False}
    ).json()["response"]

    resultado = avaliar_ensemble(
        pergunta=item["pergunta"],
        resposta=resposta,
        contexto=item["contexto"],
        esperado=item["esperado"]
    )

    for metrica, dados in resultado["ensemble"].items():
        lf.score_current_trace(
            name=f"ensemble_{metrica}",
            value=dados["score"]
        )
        for modelo, score in dados["scores_por_modelo"].items():
            lf.score_current_trace(
                name=f"{modelo}_{metrica}",
                value=score
            )

    return resultado


def verificar_threshold(resultados: list):
    totais = {k: [] for k in THRESHOLDS}
    for r in resultados:
        for metrica in THRESHOLDS:
            totais[metrica].append(r["ensemble"][metrica]["score"])

    print("\n" + "=" * 60)
    print("RESUMO FINAL — ENSEMBLE")
    print("=" * 60)
    passou = True
    for metrica, scores in totais.items():
        media = sum(scores) / len(scores)
        threshold = THRESHOLDS[metrica]
        ok = media <= threshold if metrica == "toxicity" else media >= threshold
        status = "PASSOU" if ok else "FALHOU"
        if not ok:
            passou = False
        print(f"{metrica:<15} media: {media:.2f}  threshold: {threshold}  [{status}]")

    print("=" * 60)
    print(f"RESULTADO GERAL: {'PASSOU' if passou else 'FALHOU'}")
    print("=" * 60)


if __name__ == "__main__":
    dataset = carregar_dataset()

    print("=" * 60)
    print("RODANDO ENSEMBLE DE JUDGES — llama3.2 + mistral")
    print(f"Total de itens: {len(dataset)}")
    print("=" * 60)

    resultados = []
    for item in dataset[:3]:  # roda os 3 primeiros para nao demorar muito
        resultado = avaliar_item_ensemble(item)
        imprimir_resultado(resultado)
        resultados.append(resultado)

    verificar_threshold(resultados)
    lf.flush()
