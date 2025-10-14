from dataclasses import dataclass
from typing import List


@dataclass
class QuizItem:
    pergunta: str
    alternativas: list[str]
    correta: str
    explicacao: str
