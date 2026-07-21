"""
Treina um modelo customizado dddd_trainer para o CAPTCHA do Recife em Dia.

Reaproveita a mesma engine dddd_trainer (app.py/configs/nets/utils) usada
em sefazrn e duc_sp -- inclusive as correções já validadas lá:
  - path do dddd_trainer_dir prioriza BASE_DIR/app.py
  - GPU detectada via torch.cuda.is_available() com fallback pra CPU
    embutido no utils/train.py
  - Adam + LR 0.0007 + weight_decay 0.0001 como ponto de partida
  - augmentação (rotation/affine/colorjitter/blur/gaussian noise/random
    erasing) é global em utils/load_cache.py e se aplica automaticamente

O que muda por depender do padrão visual deste captcha (5 caracteres,
MISTURA de maiúsculas/minúsculas, linha ondulada + ruído pontilhado):
  - Nome do projeto: recife
  - Pasta de dataset separada: dataset_recife/rotulado
  - Charset preserva caixa (A-Z e a-z são símbolos DIFERENTES no charset,
    não force upper/lower como foi feito no sefazrn e no duc_sp)
  - Se a acurácia sofrer por causa do ruído pontilhado de fundo (diferente
    da linha de tachado limpa do duc_sp e da imagem limpa do sefazrn), o
    ajuste a testar é aumentar RandomErasing ou o std do AddGaussianNoise
    em utils/load_cache.py -- não alterado aqui de antemão, é ajuste
    empírico.

Formato esperado (o rotular_recife.py já gera assim):
    dataset_recife/rotulado/
        Px19M_0001.png
        yqeoH_0002.png
        ...

O nome do arquivo tem o padrão  LABEL_qualquercoisa.png
O script extrai o label como tudo antes do primeiro underscore.

Saída:
    modelo/modelo_recife.onnx   ← arquivo que você usa no script principal

Uso:
    python treinar_recife.py
    python treinar_recife.py --epocas 30 --batch 64
"""

import argparse
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("treinar_recife")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR       = Path(__file__).resolve().parent
PASTA_ROTULADO = BASE_DIR / "dataset_recife" / "rotulado"
PASTA_MODELO   = BASE_DIR / "modelo"

def verificar_dependencias() -> None:
    erros = []
    try:
        import torch
        log.info("PyTorch: %s", torch.__version__)
        if torch.cuda.is_available():
            log.info("CUDA disponível: %s", torch.cuda.get_device_name(0))
        else:
            log.info("CUDA não disponível — treinando em CPU (mais lento, mas funciona)")
    except ImportError:
        erros.append("torch  →  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    try:
        import ddddocr
        log.info("ddddocr: instalado (versão não exposta pela biblioteca)")
    except ImportError:
        erros.append("ddddocr  →  pip install ddddocr")

    try:
        from PIL import Image
    except ImportError:
        erros.append("Pillow  →  pip install Pillow")

    if erros:
        print("\nDependências faltando:")
        for e in erros:
            print(f"  {e}")
        sys.exit(1)

def carregar_dataset(pasta: Path) -> tuple[list[Path], list[str], list[str]]:
    """
    Lê as imagens rotuladas e extrai os labels.

    Convenção de nome: LABEL_numero.png
    Exemplo: Px19M_0042.png  →  label = 'Px19M'

    Sem .upper()/.lower() aqui -- diferente do sefazrn (só maiúsculo) e do
    duc_sp (só minúsculo), este captcha mistura caixa e precisa preservar
    exatamente o que está desenhado.
    """
    imagens = []
    labels  = []
    charset = set()

    for arq in sorted(pasta.glob("*.png")):
        partes = arq.stem.split("_")
        label  = partes[0]

        if not re.fullmatch(r"[A-Za-z0-9]+", label):
            log.warning("Label inválido ignorado: %s (arquivo: %s)", label, arq.name)
            continue

        imagens.append(arq)
        labels.append(label)
        charset.update(label)

    return imagens, labels, sorted(charset)


def inspecionar_dataset(imagens: list, labels: list, charset: list) -> None:
    from collections import Counter

    if not imagens:
        log.error("Nenhuma imagem válida encontrada em %s", PASTA_ROTULADO)
        log.error("Rode primeiro: python rotular_recife.py")
        sys.exit(1)

    comprimentos = Counter(len(l) for l in labels)
    log.info("-" * 50)
    log.info("Dataset: %d imagens", len(imagens))
    log.info("Charset: %s (%d chars)", "".join(charset), len(charset))
    log.info("Comprimentos: %s", dict(sorted(comprimentos.items())))

    if len(imagens) < 200:
        log.warning(
            "Dataset pequeno (%d imagens). Recomendado: 300+. "
            "O modelo pode ter baixa acurácia.",
            len(imagens)
        )
    elif len(imagens) < 400:
        log.info("Dataset razoável. Para melhor resultado, tente chegar a 600+.")
    else:
        log.info("Dataset robusto! Boas chances de alta acurácia.")
    log.info("-" * 50)

def treinar(epocas: int = 50, batch: int = 64, val_split: float = 0.1) -> None:
    verificar_dependencias()

    if not PASTA_ROTULADO.exists() or not any(PASTA_ROTULADO.glob("*.png")):
        log.error("Pasta %s vazia ou inexistente.", PASTA_ROTULADO)
        log.error("Rode: python rotular_recife.py")
        sys.exit(1)

    imagens, labels, charset = carregar_dataset(PASTA_ROTULADO)
    inspecionar_dataset(imagens, labels, charset)

    PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    caminho_modelo = PASTA_MODELO / "modelo_recife.onnx"

    log.info("Iniciando treinamento: %d épocas, batch=%d", epocas, batch)

    _treinar_via_cli(imagens, labels, charset, epocas, batch, caminho_modelo)

    if caminho_modelo.exists():
        log.info("=" * 60)
        log.info("Modelo salvo em: %s", caminho_modelo)
        log.info("Próximo passo: atualize o script principal para usar o modelo customizado.")
        log.info("=" * 60)
        _mostrar_como_usar(caminho_modelo)
    else:
        log.error("Modelo não foi gerado. Verifique os erros acima.")

def _treinar_via_cli(imagens, labels, charset, epocas, batch, caminho_modelo) -> None:
    import subprocess, yaml

    python_exe = str(Path(sys.executable).parent / "python.exe")
    if not Path(python_exe).exists():
        python_exe = sys.executable
    log.info("Usando Python: %s", python_exe)

    if (BASE_DIR / "app.py").exists():
        dddd_trainer_dir = BASE_DIR
    elif (BASE_DIR / "dddd_trainer" / "app.py").exists():
        dddd_trainer_dir = BASE_DIR / "dddd_trainer"
    else:
        raise FileNotFoundError(
            f"Não encontrei app.py nem em {BASE_DIR} "
            f"nem em {BASE_DIR / 'dddd_trainer'}. Confira onde o treinar_recife.py está salvo."
        )
    projeto = "recife"
    projeto_path = dddd_trainer_dir / "projects" / projeto

    projeto_path.mkdir(parents=True, exist_ok=True)
    (projeto_path / "checkpoints").mkdir(exist_ok=True)
    (projeto_path / "models").mkdir(exist_ok=True)
    (projeto_path / "cache").mkdir(exist_ok=True)

    try:
        import torch as _torch
        gpu_disponivel = _torch.cuda.is_available()
    except ImportError:
        gpu_disponivel = False

    config = {
        "System": {
            "Project": projeto,
            "GPU": gpu_disponivel,
            "GPU_ID": 0,
            "Allow_Ext": ["jpg", "jpeg", "png", "bmp"],
            "Path": str(PASTA_ROTULADO).replace("\\", "/"),
            "Val": 0.1,
        },
        "Model": {
            "ImageWidth": -1,
            "ImageHeight": 64,
            "ImageChannel": 1,
            "CharSet": list(charset),
            "Word": False,
        },
        "Train": {
            "BATCH_SIZE": batch,
            "TEST_BATCH_SIZE": batch,
            "CNN": {"NAME": "ddddocr"},
            "DROPOUT": 0.3,
            "OPTIMIZER": "Adam",
            "TEST_STEP": 100,
            "SAVE_CHECKPOINTS_STEP": 500,
            "TARGET": {
                "Accuracy": 0.97,
                "Epoch": epocas,
                "Cost": 0.05,
            },
            "LR": 0.0007,
            "WEIGHT_DECAY": 0.0001,
        },
    }

    cfg_path = projeto_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=None, indent=4)
    log.info("Config gerada: %s", cfg_path)

    log.info("Cacheando dataset...")
    subprocess.run(
        [python_exe, "app.py", "cache", projeto, str(PASTA_ROTULADO), "--search_type=name"],
        check=True, cwd=str(dddd_trainer_dir)
    )

    log.info("Iniciando treino via dddd_trainer...")
    subprocess.run(
        [python_exe, "app.py", "train", projeto],
        check=True, cwd=str(dddd_trainer_dir)
    )

    modelos = sorted((projeto_path / "models").glob("*.onnx"))
    if modelos:
        import shutil
        shutil.copy(modelos[-1], caminho_modelo)
        log.info("Modelo copiado: %s → %s", modelos[-1], caminho_modelo)
    else:
        log.error("Nenhum .onnx encontrado em %s", projeto_path / "models")

def _mostrar_como_usar(caminho_modelo: Path) -> None:
    print("\n" + "=" * 60)
    print("Como usar o modelo treinado no script principal:")
    print("=" * 60)
    print(f"""
import ddddocr

# Substitua a instanciação padrão por:
ocr = ddddocr.DdddOcr(
    show_ad=False,
    import_onnx_path=r"{caminho_modelo}",
    charsets_path=r"{caminho_modelo.parent / 'charsets.json'}",
)

# Uso idêntico ao original:
resultado = ocr.classification(png_bytes)
""")
    print("=" * 60)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Treina modelo dddd_trainer para CAPTCHA Recife em Dia")
    parser.add_argument("--epocas", type=int, default=50,   help="Número de épocas (padrão: 50)")
    parser.add_argument("--batch",  type=int, default=64,   help="Tamanho do batch (padrão: 64)")
    args = parser.parse_args()

    treinar(epocas=args.epocas, batch=args.batch)