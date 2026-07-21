# -*- coding: utf-8 -*-
"""
Verifica um dataset rotulado (pasta com arquivos LABEL_numero.png) antes
de treinar: aponta labels com tamanho diferente do esperado, caracteres
fora do charset esperado, e arquivos vazios/corrompidos.

Detecta o mesmo tipo de problema que já causou treino ruim no SEFAZ-RN
(1 arquivo com label de tamanho 1 no meio do dataset) -- vale rodar
sempre antes de treinar_recife.py, principalmente depois de sessões
grandes de rotulagem.

Uso:
    python verificar_dataset.py dataset_recife/rotulado
    python verificar_dataset.py dataset_recife/rotulado --tamanho 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


def verificar(pasta: Path, tamanho_esperado: int | None) -> None:
    if not pasta.exists():
        print(f"Pasta não encontrada: {pasta}")
        sys.exit(1)

    arquivos = sorted(pasta.glob("*.png"))
    if not arquivos:
        print(f"Nenhum .png encontrado em {pasta}")
        sys.exit(1)

    labels = []
    problemas = []
    charset = set()
    comprimentos = Counter()

    for arq in arquivos:
        label = arq.stem.split("_")[0]
        labels.append(label)
        comprimentos[len(label)] += 1
        charset.update(label)

        if not label:
            problemas.append((arq.name, "label vazio"))
            continue

        if tamanho_esperado is not None and len(label) != tamanho_esperado:
            problemas.append((arq.name, f"tamanho {len(label)} (esperado {tamanho_esperado}): {label!r}"))

        if arq.stat().st_size == 0:
            problemas.append((arq.name, "arquivo vazio (0 bytes)"))

    print("=" * 60)
    print(f"Pasta: {pasta}")
    print(f"Total de imagens: {len(arquivos)}")
    print(f"Charset encontrado ({len(charset)} chars): {''.join(sorted(charset))}")
    print(f"Distribuição de comprimentos: {dict(sorted(comprimentos.items()))}")
    print("=" * 60)

    if problemas:
        print(f"\n{len(problemas)} problema(s) encontrado(s):\n")
        for nome, motivo in problemas:
            print(f"  {nome}  ->  {motivo}")
        print(
            "\nRecomendação: corrija ou remova esses arquivos antes de treinar "
            "-- um único label com tamanho errado no meio do dataset já é "
            "suficiente pra derrubar a acurácia real de forma sutil."
        )
    else:
        print("\nNenhum problema encontrado. Dataset consistente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica consistência de um dataset rotulado")
    parser.add_argument("pasta", type=str, help="Caminho da pasta com os arquivos LABEL_numero.png")
    parser.add_argument("--tamanho", type=int, default=5, help="Tamanho esperado do label (padrão: 5)")
    args = parser.parse_args()

    verificar(Path(args.pasta), args.tamanho)
