# -*- coding: utf-8 -*-
"""
Ferramenta de rotulagem interativa de CAPTCHAs — Recife em Dia.

Diferenças que dependem do padrão visual deste captcha (5 caracteres,
MISTURA de maiúsculas e minúsculas, linha ondulada + pontilhado de ruído
sobre o texto -- ex.: "8oToL", "Px19M", "yqeoH"):
  1. TAM_CAPTCHA = 5.
  2. Label NÃO é forçado nem para maiúsculo nem para minúsculo -- salva
     exatamente o que for digitado, preservando a caixa de cada letra.

Atalhos:
    Enter     — confirma o label e avança
    Ctrl+S    — pula esta imagem
    Ctrl+D    — descarta esta imagem
    Esc       — sai e salva o progresso

Uso:
    python rotular_recife.py
"""

import shutil
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

BASE_DIR       = Path(__file__).resolve().parent
PASTA_RAW      = BASE_DIR / "dataset_recife" / "raw"
PASTA_ROTULADO = BASE_DIR / "dataset_recife" / "rotulado"
PASTA_PULADOS  = BASE_DIR / "dataset_recife" / "pulados"
PASTA_DESCARTE = BASE_DIR / "dataset_recife" / "descartado"

# AJUSTE: captcha do Recife em Dia tem 5 caracteres, mistura maiúsc/minúsc.
TAM_CAPTCHA = 5
ZOOM = 5

CHARS_VALIDOS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def ja_rotuladas() -> set[str]:
    stems = set()
    for pasta in [PASTA_ROTULADO, PASTA_PULADOS, PASTA_DESCARTE]:
        if pasta.exists():
            stems.update(p.stem for p in pasta.glob("*.png"))
    return stems


def coletar_pendentes() -> list[Path]:
    if not PASTA_RAW.exists() or not any(PASTA_RAW.glob("*.png")):
        print("Pasta dataset_recife/raw não encontrada ou vazia. Rode coletar_recife.py primeiro.")
        sys.exit(1)
    todas     = sorted(PASTA_RAW.glob("*.png"))
    rotuladas = ja_rotuladas()
    return [p for p in todas if p.stem not in rotuladas]


class Rotulador(tk.Tk):
    def __init__(self, pendentes: list[Path]):
        super().__init__()
        self.title("Rotulador de Captchas — Recife em Dia")
        self.resizable(False, False)

        for pasta in [PASTA_ROTULADO, PASTA_PULADOS, PASTA_DESCARTE]:
            pasta.mkdir(parents=True, exist_ok=True)

        self.pendentes = pendentes
        self.total     = len(pendentes)
        self.idx       = 0
        self.rotuladas_agora = 0
        self.puladas         = 0
        self.descartadas     = 0

        self.lbl_progresso = tk.Label(self, font=("Segoe UI", 11))
        self.lbl_progresso.pack(pady=(10, 4))

        self.lbl_imagem = tk.Label(self)
        self.lbl_imagem.pack(pady=8)

        self.lbl_instrucao = tk.Label(
            self,
            text="Digite o label (respeite maiúsc/minúsc) e Enter  |  Ctrl+S pular  |  Ctrl+D descartar  |  Esc sair",
            font=("Segoe UI", 9), fg="#555",
        )
        self.lbl_instrucao.pack(pady=(0, 4))

        self.entrada = tk.Entry(self, font=("Consolas", 20), justify="center", width=12)
        self.entrada.pack(pady=(0, 12))
        self.entrada.bind("<Return>", self.confirmar)
        self.entrada.bind("<Control-s>", self.pular)
        self.entrada.bind("<Control-d>", self.descartar)
        self.bind("<Escape>", self.sair)

        self.lbl_status = tk.Label(self, font=("Segoe UI", 9), fg="#a00")
        self.lbl_status.pack(pady=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self.sair)
        self.mostrar_atual()
        self.entrada.focus_set()

    def mostrar_atual(self):
        if self.idx >= self.total:
            self.finalizar()
            return

        caminho = self.pendentes[self.idx]
        self.caminho_atual = caminho

        img = Image.open(caminho)
        img_grande = img.resize((img.width * ZOOM, img.height * ZOOM), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(img_grande)
        self.lbl_imagem.configure(image=self.tk_img)

        self.lbl_progresso.configure(
            text=f"[{self.idx + 1}/{self.total}] {caminho.name}"
        )
        self.lbl_status.configure(text="")
        self.entrada.delete(0, tk.END)
        self.entrada.focus_set()

    def confirmar(self, event=None):
        # Sem .upper() e sem .lower() -- este captcha mistura caixa e o
        # label salvo precisa refletir exatamente o que está desenhado.
        texto = self.entrada.get().strip()
        invalidos = [c for c in texto if c not in CHARS_VALIDOS]

        if invalidos:
            self.lbl_status.configure(text=f"Caracteres inválidos: {invalidos}")
            return

        if not texto:
            self.lbl_status.configure(text="Digite um label (ou Ctrl+S para pular).")
            return

        if len(texto) != TAM_CAPTCHA:
            self.lbl_status.configure(
                text=f"Aviso: comprimento {len(texto)} (esperado {TAM_CAPTCHA}). Enter de novo para forçar."
            )
            if not getattr(self, "_confirmar_len_diferente", False):
                self._confirmar_len_diferente = True
                return
            self._confirmar_len_diferente = False

        numero  = self.caminho_atual.stem
        # Nome de arquivo case-sensitive: no Windows/NTFS o filesystem não
        # diferencia caixa por padrão, mas como o "_" separa label do
        # número, duas labels com a mesma grafia salva-se sem conflito
        # de verdade (o número garante unicidade); mantido por consistência
        # com os outros projetos do repo.
        destino = PASTA_ROTULADO / f"{texto}_{numero}.png"
        if destino.exists():
            import time
            destino = PASTA_ROTULADO / f"{texto}_{numero}_{int(time.time())}.png"

        shutil.move(str(self.caminho_atual), str(destino))
        self.rotuladas_agora += 1
        self.idx += 1
        self.mostrar_atual()

    def pular(self, event=None):
        destino = PASTA_PULADOS / self.caminho_atual.name
        shutil.move(str(self.caminho_atual), str(destino))
        self.puladas += 1
        self.idx += 1
        self.mostrar_atual()

    def descartar(self, event=None):
        destino = PASTA_DESCARTE / self.caminho_atual.name
        shutil.move(str(self.caminho_atual), str(destino))
        self.descartadas += 1
        self.idx += 1
        self.mostrar_atual()

    def sair(self, event=None):
        self.finalizar()

    def finalizar(self):
        total_rotulado = len(list(PASTA_ROTULADO.glob("*.png"))) if PASTA_ROTULADO.exists() else 0
        print("\n" + "=" * 60)
        print(f"  Nesta sessão: {self.rotuladas_agora} rotuladas | {self.puladas} puladas | {self.descartadas} descartadas")
        print(f"  Total rotulado acumulado: {total_rotulado} imagens")
        if total_rotulado < 500:
            print(f"  Ainda faltam ~{500 - total_rotulado} imagens para um bom dataset inicial.")
        else:
            print(f"  Excelente! Você já pode treinar o dddd_trainer com confiança.")
        print("=" * 60)
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    pendentes = coletar_pendentes()

    print("=" * 60)
    print(f"  Encontradas {len(pendentes)} imagens prontas para rotular")
    print("=" * 60)

    app = Rotulador(pendentes)
    app.mainloop()