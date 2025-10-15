from src.ftc4.core.config import settings
from dataclasses import asdict
import json
import os

def main():
    """
    Função para verificar a carga das variaveis de projeto e ambiente (PATH)
    Já está funcional mas falta incluir os logs para ficar completo.
    """

    # Converte a instância para um dicionário (dict)
    settings_dict = asdict(settings)

    # Expande e printa a variavel PATH
    print('\n')
    print("-" * 25, "   PATH   ", "-"*25)
    for i in os.getenv("PATH").split(os.pathsep):
        print(i)

    # Printa o dicionário formatado
    print('\n')
    print("-" * 25, "   PROJECT   ", "-"*25)
    print(json.dumps(settings_dict, indent=4, default=str))


if __name__ == "__main__":
    main()