import torch
def main():
    """
    Função para verificar qual o dispositivo está sendo usado como motor principal cpu/gpu
    Já está funcional mas falta incluir os logs para ficar completo.
    """
    print("=" * 50)
    print(f"Versão do PyTorch: {torch.__version__}")

    # Verifica se CUDA (GPU da NVIDIA) está disponível
    if torch.cuda.is_available():
        print("CUDA está disponível!")
        print(f"Dispositivo atual: {torch.cuda.get_device_name(0)}")
        print(f"Quantidade de GPUs detectadas: {torch.cuda.device_count()}")
        print(f"Versão do CUDA: {torch.version.cuda}")
    else:
        print("CUDA não está disponível. PyTorch está usando CPU.")
        print("Dispositivo atual: CPU")

    print("=" * 50)

if __name__ == "__main__":
    main()