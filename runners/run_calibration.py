import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from judges import relevancy, faithfulness

try:
    from sklearn.metrics import confusion_matrix, cohen_kappa_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("Aviso: sklearn nao instalado. Rode: pip install scikit-learn")


MODELO = "llama3.2"


def carregar_anotacoes() -> list:
    caminho = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "human_annotations.json"
    )
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def binarizar(score: float, threshold: float) -> int:
    return 1 if score >= threshold else 0


def calcular_metricas(humanos: list, judge: list, nome: str):
    print(f"\n--- {nome} ---")
    if not SKLEARN_OK:
        concordancia = sum(h == j for h, j in zip(humanos, judge)) / len(humanos)
        print(f"  Concordancia simples: {concordancia:.2%}")
        return

    tn, fp, fn, tp = confusion_matrix(humanos, judge).ravel() if len(set(humanos)) > 1 else (0, 0, 0, 0)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    kappa = cohen_kappa_score(humanos, judge) if len(set(humanos)) > 1 else 0

    status_tpr   = "PASSOU" if tpr   >= 0.9  else "FALHOU"
    status_tnr   = "PASSOU" if tnr   >= 0.9  else "FALHOU"
    status_kappa = "PASSOU" if kappa >= 0.7  else "FALHOU"

    print(f"  TPR (detecta passes):  {tpr:.2%}  [{status_tpr}]  (meta: > 90%)")
    print(f"  TNR (detecta falhas):  {tnr:.2%}  [{status_tnr}]  (meta: > 90%)")
    print(f"  Kappa de Cohen:        {kappa:.2f}   [{status_kappa}]  (meta: > 0.7)")


if __name__ == "__main__":
    anotacoes = carregar_anotacoes()

    print("=" * 60)
    print("CALIBRACAO DO JUDGE")
    print(f"Total de amostras: {len(anotacoes)}")
    print("=" * 60)

    humanos_relevancy    = []
    judge_relevancy      = []
    humanos_faithfulness = []
    judge_faithfulness   = []

    for item in anotacoes:
        pergunta = item["pergunta"]
        resposta = item["resposta_modelo"]
        anotacao = item["anotacao_humana"]

        print(f"\nAvaliando: {pergunta[:50]}...")

        r_rel = relevancy.avaliar(pergunta, resposta, MODELO)
        r_fai = faithfulness.avaliar(pergunta, resposta, MODELO)

        humanos_relevancy.append(anotacao["relevancy"])
        judge_relevancy.append(binarizar(r_rel["score"], 0.7))

        humanos_faithfulness.append(anotacao["faithfulness"])
        judge_faithfulness.append(binarizar(r_fai["score"], 0.7))

        print(f"  Humano relevancy:    {anotacao['relevancy']} | Judge: {r_rel['score']} ({r_rel['justificativa']})")
        print(f"  Humano faithfulness: {anotacao['faithfulness']} | Judge: {r_fai['score']} ({r_fai['justificativa']})")

    print("\n" + "=" * 60)
    print("RESULTADO DA CALIBRACAO")
    print("=" * 60)
    calcular_metricas(humanos_relevancy,    judge_relevancy,    "RELEVANCY")
    calcular_metricas(humanos_faithfulness, judge_faithfulness, "FAITHFULNESS")
    print("\nCalibracao concluida. Ajuste os prompts dos judges se TPR/TNR < 90%.")
