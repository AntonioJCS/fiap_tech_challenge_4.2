from __future__ import annotations

def safe_int(value) -> int | None:
    """
    Tenta converter um valor para inteiro, retornando o valor se for
    bem-sucedido ou None em caso de falha ou se o valor for None.
    """
    try:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        # Tenta converter string para int
        return int(str(value).strip())
    except Exception:
        return None

def safe_float(value) -> float | None:
    """
    Tenta converter um valor para float, retornando o valor se for
    bem-sucedido ou None em caso de falha ou se o valor for None.
    """
    try:
        if value is None:
            return None
        # Se já for int ou float, converte para float para normalizar
        if isinstance(value, (float, int)):
            return float(value)
        # Tenta converter string para float, tratando vírgula como separador decimal
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return None

def safe_int_default(value, default: int) -> int:
    """
    Tenta converter um valor para inteiro, retornando o valor se for
    bem-sucedido ou Default em caso de falha ou se o valor for None.
    """
    try:
        if value is None:
            return default
        if isinstance(value, (int,)):
            return value
        return int(str(value).strip())
    except Exception:
        return default

def safe_float_default(value, default: float) -> float:
    """
    Tenta converter um valor para Float, retornando o valor se for
    bem-sucedido ou Default em caso de falha ou se o valor for None.
    """
    try:
        if value is None:
            return default
        if isinstance(value, (float, int)):
            return float(value)
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return default

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def clamp_int(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))
