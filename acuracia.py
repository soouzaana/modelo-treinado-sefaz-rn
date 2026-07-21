"""
Mede a acurácia real do modelo (comparando a string inteira decodificada
contra o label) separadamente em TREINO e em VALIDAÇÃO -- usando os MESMOS
arquivos de cache (cache.train.tmp / cache.val.tmp) e o MESMO pré-processamento
(resize proporcional + Normalize) que o próprio dddd_trainer usa durante o
treino. Isso evita qualquer mismatch de pré-processamento entre teste e
treino, e mostra separado se o problema é overfitting (treino alto, val
baixo) ou algo mais estrutural (os dois baixos).

Roda em model.eval() (running stats do BatchNorm) por padrão -- é o modo
que reflete o que vai pro ONNX/produção. Passe --train-mode pra comparar
também com model.train() (batch stats), útil pra confirmar se a
recalibração de BN funcionou.

Uso:
    python medir_acuracia_val.py <project_name> <checkpoint.tar> [--train-mode]

Exemplo:
    python medir_acuracia_val.py sefaz_ms checkpoint_sefaz_ms_5185_140000_bncalib.tar --train-mode
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import Config
from nets import Net
from utils.load_cache import GetLoader, LoadCache


def build_full_loader(get_loader: GetLoader, cache_path: str, augment: bool):
    """Recria um DataLoader cobrindo TODO o cache (sem drop_last, sem shuffle),
    usando o mesmo pré-processamento (val ou train) já montado em GetLoader."""
    dataset = LoadCache(cache_path, get_loader.path, get_loader.word,
                        get_loader.ImageChannel, get_loader.resize, get_loader.charset)
    collate = lambda batch: get_loader.collate_to_sparse(batch, augment=augment)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset),
                                       shuffle=False, drop_last=False,
                                       num_workers=0, collate_fn=collate)


def avaliar(net, loader, charset, label="conjunto"):
    total_correct = 0
    total = 0
    erros_mostrados = 0
    with torch.no_grad():
        for images, values, shapes in loader:
            pred_decode_labels, labels_list, correct_list, error_list = net.tester(images, values, shapes)
            total_correct += len(correct_list)
            total += len(labels_list)
            for idx in error_list:
                if erros_mostrados < 15:
                    pred_str = "".join(charset[int(i)] for i in pred_decode_labels[idx])
                    label_str = "".join(charset[int(i)] for i in labels_list[idx])
                    print(f"    errado: label={label_str!r:10s} pred={pred_str!r}")
                    erros_mostrados += 1
    acc = total_correct / total if total else 0.0
    print(f"  {label}: {total_correct}/{total} corretos ({acc*100:.1f}%)")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name")
    parser.add_argument("checkpoint_name")
    parser.add_argument("--train-mode", action="store_true",
                        help="Também mede em model.train() (batch stats) pra comparar com eval()")
    args = parser.parse_args()

    device = torch.device("cpu")
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects", args.project_name)
    ckpt_path = os.path.join(project_path, "checkpoints", args.checkpoint_name)
    if not os.path.exists(ckpt_path):
        print(f"Erro: checkpoint não encontrado em {ckpt_path}")
        return

    conf = Config(args.project_name).load_config()
    charset = conf["Model"]["CharSet"]
    net = Net(conf, lr=conf["Train"]["LR"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint["net"])
    print(f"Checkpoint: epoch={checkpoint.get('epoch')} step={checkpoint.get('step')}\n")

    get_loader = GetLoader(args.project_name)
    train_loader = build_full_loader(get_loader, get_loader.cache_train_path, augment=False)
    val_loader = build_full_loader(get_loader, get_loader.cache_val_path, augment=False)

    print("=== model.eval() (running stats -- é o que vai pro ONNX/produção) ===")
    net.eval()
    avaliar(net, train_loader, charset, "TREINO")
    avaliar(net, val_loader, charset, "VALIDAÇÃO")

    if args.train_mode:
        print("\n=== model.train() (batch stats -- só diagnóstico, não usar em produção) ===")
        net.train()
        avaliar(net, train_loader, charset, "TREINO")
        avaliar(net, val_loader, charset, "VALIDAÇÃO")
        net.eval()


if __name__ == "__main__":
    main()