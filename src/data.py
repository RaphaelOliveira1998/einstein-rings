"""Carregamento e preparação dos dados de detecção de lentes gravitacionais.

Dataset DeepLense lens finding (HSC-SSP): um ``.npy`` por objeto, shape
``(3, 64, 64)``, bandas g, r, i em ``float32`` normalizado em ``[0, 1]``.

Este módulo é a única fonte de verdade sobre onde os dados moram e como são
lidos. Notebooks e scripts de treino importam daqui em vez de repetir paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Raiz do repo = pasta pai de src/. Independe de onde o processo foi iniciado.
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.environ.get("EINSTEIN_RAW_DIR", REPO_ROOT / "data" / "raw"))

# Rótulo por classe: lente = 1 (positivo, o que queremos achar), não-lente = 0.
CLASS_TO_LABEL = {"lenses": 1, "nonlenses": 0}
BANDS = ("g", "r", "i")
IMG_SHAPE = (3, 64, 64)
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fixa a seed do numpy (e do torch, se disponível) para reprodutibilidade."""
    np.random.seed(seed)
    try:  # torch é opcional, só faz falta no treino
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Listagem
# ---------------------------------------------------------------------------

def list_split(split: str, raw_dir: Path | str | None = None):
    """Lista os arquivos de um split e seus rótulos.

    Parameters
    ----------
    split : {"train", "test"}
    raw_dir : path opcional; por padrão :data:`RAW_DIR`.

    Returns
    -------
    paths : np.ndarray[str]  : caminhos dos ``.npy``
    labels : np.ndarray[int] : 1 = lente, 0 = não-lente
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    paths, labels = [], []
    for cls, label in CLASS_TO_LABEL.items():
        folder = raw_dir / f"{split}_{cls}"
        if not folder.is_dir():
            raise FileNotFoundError(f"pasta esperada não existe: {folder}")
        # ordena por índice numérico do nome (1.npy, 2.npy, ...) p/ estabilidade
        files = sorted(folder.glob("*.npy"), key=lambda p: int(p.stem))
        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))
    return np.array(paths), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Carrega um objeto como ``float32`` de shape ``(3, 64, 64)``.

    Valida o shape na fronteira com o disco: um arquivo corrompido falha aqui,
    com o nome do arquivo, em vez de estourar dentro da rede várias camadas
    adiante.
    """
    img = np.load(path).astype(np.float32)
    if img.shape != IMG_SHAPE:
        raise ValueError(f"shape inesperado {img.shape} em {path}")
    return img
