# -*- coding: utf-8 -*-
"""
Coleta imagens de CAPTCHA do portal Recife em Dia (Emissão de Certidão)
para rotulagem manual.

Terceiro padrão de captcha do repo (depois de sefazrn e duc_sp). Diferença
de arquitetura: aqui é PHP puro (cookie PHPSESSID), endpoint
/captcha/showImage?namespace=...&<token> -- devolve a imagem crua no corpo
(sem JSON/base64), igual ao duc_sp. O <token> depois do & é só cache-buster
(no HTML inicial vem um float tipo "0.608107461176657"; numa recarga via
JS observada no DevTools veio um hex de 32 chars -- ou seja, o valor em si
não importa pro servidor, só precisa ser diferente a cada request pra não
tomar cache).

Salva cada imagem como:
    dataset_recife/raw/<NUMERO>.png

Uso:
    python coletar_recife.py          # coleta 300 imagens (padrão)
    python coletar_recife.py 500      # coleta 500 imagens
"""

import io
import logging
import secrets
import sys
import time
from pathlib import Path

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL       = "https://recifeemdia.recife.pe.gov.br"
URL_FORMULARIO = f"{BASE_URL}/emissaoCertidao/4"
URL_CAPTCHA    = f"{BASE_URL}/captcha/showImage"
NAMESPACE      = "formulario_emissao_pdf"

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

PAUSA = 1.2   # segundos entre requisições (seja gentil com o servidor)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coletar_recife")

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Funções de rede / imagem
# ---------------------------------------------------------------------------

def obter_pagina_inicial(session: requests.Session) -> None:
    """
    Acessa a página do formulário uma vez para que o servidor estabeleça
    o cookie de sessão (PHPSESSID) antes de pedir a imagem do captcha.
    """
    try:
        session.get(
            URL_FORMULARIO,
            headers=HEADERS_BASE,
            timeout=30,
        )
    except requests.RequestException as exc:
        log.warning("Não foi possível acessar a página inicial: %s", exc)


def buscar_captcha_bytes(session: requests.Session) -> bytes:
    """
    Cada GET em /captcha/showImage gera um captcha novo associado à sessão
    atual e devolve a imagem crua (sem JSON, sem base64). O token depois
    de "&" é só cache-buster -- usa um hex aleatório, mas qualquer valor
    diferente a cada chamada serve.
    """
    token = secrets.token_hex(16)
    url = f"{URL_CAPTCHA}?namespace={NAMESPACE}&{token}"
    resp = session.get(
        url,
        headers={
            **HEADERS_BASE,
            "Accept":  "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": URL_FORMULARIO,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def processar_imagem_captcha(raw_bytes: bytes) -> bytes:
    """Converte os bytes crus da imagem para PNG em escala de cinza."""
    img_orig = Image.open(io.BytesIO(raw_bytes))
    img_orig.load()
    img_final = img_orig.convert("L")

    buffer = io.BytesIO()
    img_final.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Coleta principal
# ---------------------------------------------------------------------------

def coletar(n: int = 300) -> None:
    pasta_raw = BASE_DIR / "dataset_recife" / "raw"
    pasta_raw.mkdir(parents=True, exist_ok=True)

    existentes = sorted(pasta_raw.glob("*.png"))
    proximo    = len(existentes) + 1

    session = requests.Session()
    session.headers.update({"User-Agent": HEADERS_BASE["User-Agent"]})

    log.info("Inicializando sessão...")
    obter_pagina_inicial(session)

    log.info("Coletando %d imagens (a partir de #%04d)...", n, proximo)
    log.info("Pasta de saída: %s", pasta_raw)

    salvos   = 0
    erros    = 0
    i        = proximo

    RENOVAR_SESSAO_A_CADA = 50

    while salvos < n:
        try:
            if salvos > 0 and salvos % RENOVAR_SESSAO_A_CADA == 0:
                log.info("Renovando sessão após %d captchas...", salvos)
                session = requests.Session()
                session.headers.update({"User-Agent": HEADERS_BASE["User-Agent"]})
                obter_pagina_inicial(session)

            raw = buscar_captcha_bytes(session)
            png = processar_imagem_captcha(raw)

            caminho = pasta_raw / f"{i:04d}.png"
            caminho.write_bytes(png)

            salvos += 1
            i      += 1

            if salvos % 25 == 0 or salvos == n:
                log.info("Progresso: %d/%d imagens salvas", salvos, n)

        except requests.RequestException as exc:
            erros += 1
            log.warning("Erro de rede (#%d): %s", i, exc)
            if erros > 10:
                log.error("Muitos erros consecutivos. Abortando.")
                break
            time.sleep(5)
            continue
        except Exception as exc:
            log.error("Erro inesperado (imagem inválida?): %s", exc, exc_info=True)
            erros += 1

        time.sleep(PAUSA)

    log.info("=" * 60)
    log.info("Coleta concluída: %d imagens salvas, %d erros", salvos, erros)
    log.info("Próximo passo: rode  python rotular_recife.py  para rotular as imagens")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    n = 300
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Argumento inválido: '{sys.argv[1]}'. Usando padrão (300).")

    coletar(n)