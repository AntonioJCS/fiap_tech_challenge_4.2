from src.nlp.generate import generate_outputs


def test_generate_minimal():
    text = "Este é um conteúdo simples sobre aprendizado de máquina e redes neurais."
    outs = generate_outputs(text, n_questions=5)
    assert isinstance(outs.fichamento_md, str)
    assert isinstance(outs.quiz_json, list)
