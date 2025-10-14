"""
Heurística simples de avaliação do quiz:
- verifica campos obrigatórios
- diversidade de alternativas
- taxa de preenchimento
"""

import json, sys, glob


def load(path):
    return json.loads(open(path, encoding="utf-8").read())


def score_quiz(q):
    ok = 0
    for it in q:
        if all(k in it for k in ["pergunta", "alternativas", "correta", "explicacao"]):
            if isinstance(it["alternativas"], list) and len(it["alternativas"]) >= 4:
                ok += 1
    return ok / max(len(q), 1)


def main():
    paths = sys.argv[1:] or glob.glob("data/outputs/quiz_*.json")
    for p in paths:
        q = load(p)
        print(p, "score=", round(score_quiz(q), 2))


if __name__ == "__main__":
    main()
