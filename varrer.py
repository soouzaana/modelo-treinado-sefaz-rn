"""
Varre TODOS os checkpoints salvos de um projeto, mede a acurácia real
(string inteira, model.eval()) no conjunto de VALIDAÇÃO de cada um, e
mostra qual foi o melhor -- útil pra "early stopping retroativo": muitas
vezes o checkpoint mais recente já passou do ponto ideal e está mais
decorado (overfit) do que os anteriores.

Reaproveita a mesma lógica de carregamento/decodificação do
medir_acuracia_val.py (mesmo pré-processamento usado no treino).

Uso:
    python varrer_checkpoints.py <project_name> [--every N] [--top K]

    --every N   avalia só 1 a cada N checkpoints (default: 1, avalia todos;
                útil se tiver muitos checkpoints salvos e quiser ser mais rápido)
    --top K     mostra os K melhores no final (default: 5)

Exemplo:
    python varrer_checkpoints.py sefazrn --every 2 --top 5
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import Config
from nets import Net
from utils.load_cache import GetLoader, LoadCache


def build_full_loader(get_loader: GetLoader, cache_path: str):
    dataset = LoadCache(cache_path, get_loader.path, get_loader.word,
                        get_loader.ImageChannel, get_loader.resize, get_loader.charset)
    collate = lambda batch: get_loader.collate_to_sparse(batch, augment=False)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset),
                                       shuffle=False, drop_last=False,
                                       num_workers=0, collate_fn=collate)


def medir(net, loader):
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, values, shapes in loader:
            _pred, labels_list, correct_list, _err = net.tester(images, values, shapes)
            total_correct += len(correct_list)
            total += len(labels_list)
    return total_correct, total


def checkpoint_sort_key(fname):
    # nome esperado: checkpoint_<project>_<epoch>_<step>.tar
    m = re.search(r"_(\d+)_(\d+)\.tar$", fname)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))  # ordena por epoch, depois step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name")
    parser.add_argument("--every", type=int, default=1,
                        help="Avalia 1 a cada N checkpoints (default: 1 = todos)")
    parser.add_argument("--top", type=int, default=5,
                        help="Quantos melhores mostrar no resumo final (default: 5)")
    args = parser.parse_args()

    device = torch.device("cpu")
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects", args.project_name)
    ckpt_dir = os.path.join(project_path, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"Erro: pasta de checkpoints não encontrada em {ckpt_dir}")
        return

    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".tar") and not f.endswith("_bncalib.tar")]
    checkpoints.sort(key=checkpoint_sort_key)
    if not checkpoints:
        print("Nenhum checkpoint encontrado.")
        return

    checkpoints = checkpoints[::args.every]
    print(f"Avaliando {len(checkpoints)} checkpoint(s)...\n")

    conf = Config(args.project_name).load_config()
    net = Net(conf, lr=conf["Train"]["LR"])
    net.eval()

    get_loader = GetLoader(args.project_name)
    val_loader = build_full_loader(get_loader, get_loader.cache_val_path)
    train_loader = build_full_loader(get_loader, get_loader.cache_train_path)

    resultados = []
    for fname in checkpoints:
        ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location=device)
        net.load_state_dict(ckpt["net"])
        net.eval()

        val_correct, val_total = medir(net, val_loader)
        train_correct, train_total = medir(net, train_loader)
        val_acc = val_correct / val_total if val_total else 0.0
        train_acc = train_correct / train_total if train_total else 0.0

        epoch = ckpt.get("epoch")
        step = ckpt.get("step")
        print(f"{fname:45s} epoch={epoch:<6} step={step:<7} "
              f"TREINO={train_acc*100:5.1f}%  VAL={val_acc*100:5.1f}% ({val_correct}/{val_total})")

        resultados.append((fname, epoch, step, train_acc, val_acc))

    print("\n" + "=" * 78)
    print(f"TOP {args.top} por acurácia de VALIDAÇÃO:")
    print("=" * 78)
    melhores = sorted(resultados, key=lambda r: r[4], reverse=True)[:args.top]
    for fname, epoch, step, train_acc, val_acc in melhores:
        print(f"  {fname:45s} epoch={epoch:<6} step={step:<7} "
              f"TREINO={train_acc*100:5.1f}%  VAL={val_acc*100:5.1f}%")

    if melhores:
        print(f"\n>>> Melhor checkpoint: {melhores[0][0]} (VAL={melhores[0][4]*100:.1f}%)")


if __name__ == "__main__":
    main()