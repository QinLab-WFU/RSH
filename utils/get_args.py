import argparse
import math
import os

import xlrd


def get_args():

    parser = argparse.ArgumentParser()
    # archive_v_DECH,ppo_triplet,ppo_m
    parser.add_argument("--hash-layer", type=str, default="linear", help="choice a hash layer [select, linear] to run. select: select mechaism, linear: sign function.")
    parser.add_argument("--save-dir", type=str, default="./result/")
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--dataset", type=str, default="archive_v", help="choise from [coco, archive_v, nuswide, IAPR, imagenet]")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")

    parser.add_argument("--output-dim", type=int, default=16)

    parser.add_argument("--HM", type=int, default=500)
    # parser.add_argument("--margin", type=int, default=0.1)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--alpha", type=int, default=1)

    parser.add_argument("--tau", type=int, default=0.1)

    parser.add_argument("--epochs", type=int, default=101)
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--query-num", type=int, default=5000)
    # ppo设置为：12000
    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1814)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--n_classes", type=int, default=21, help="number of dataset classes")

    parser.add_argument("--margin", type=float, default=1.0, help="None")
    parser.add_argument("--scaling_p", type=float, default=1.0, help="None")
    parser.add_argument("--scaling_x", type=float, default=3.0, help="None")
    parser.add_argument("--h_dim", type=int, default=0, help="None")
    parser.add_argument("--prior", type=int, default=0, help="None")
    parser.add_argument("--eta", type=float, default=100, help="None")

    parser.add_argument("--eta_e", type=float, default=0.01, help="None")

    parser.add_argument("--smooth_factor", type=float, default=0.15, help="None")
    parser.add_argument("--lam", type=float, default=0.01, help="None")
    parser.add_argument("--m", type=float, default=1, help="None")
    parser.add_argument("--model_lr", type=float, default=0.0001, help="None")
    parser.add_argument("--proxy_nca_lr", type=float, default=0.015, help="None")
    parser.add_argument("--embedding_lr", type=float, default=0.001, help="None")

    parser.add_argument("--trainer", type=str, default="ProgCoPL", help="name of trainer")

    # 新增蒸馏相关参数
    parser.add_argument('--K', type=int, default=4, help='因子化子模块数量')
    parser.add_argument('--distill_weight', type=float, default=0.5, help='蒸馏损失权重')

    parser.add_argument("--is-train", action="store_true" ,default=True)

    parser.add_argument('--loss', default='MS', type=str)

    parser.add_argument("--noise-rate", type=float, default=0.0)  # noise 0.2 0.5 0.8

    # IDML的参数：
    # special settings
    parser.add_argument("--loss_IDML", type=str, default="pnca", help="panc/pnca")

    parser.add_argument("--alpha_IDML", type=int, default=32, help="hyper-parameter for ProxyAnchor")
    parser.add_argument("--delta_IDML", type=float, default=0.1, help="hyper-parameter for ProxyAnchor")
    parser.add_argument("--scale_IDML", type=int, default=1, help="hyper-parameter for ProxyNCA")

    # 两阶段训练参数
    parser.add_argument('--phase1_epochs', default=30, type=int, help='第一阶段训练轮数')
    parser.add_argument('--lambda1', default=1.0, type=float, help='对比损失权重')
    parser.add_argument('--lambda2', default=10.0, type=float, help='相似性保持损失权重')

    # tau1
    parser.add_argument('--tau1', default=9.0, type=float, help='RSH的温度参数')
    parser.add_argument('--gamma1', default=0.0, type=float, help='RSH的温度参数')
    # parser.add_argument("--is-train", action="store_true")

    # 是否在验证（validation）阶段开启图像退化 + mCE 评估（ImageNet-C 风格）
    parser.add_argument(
        '--eval-corruption',
        # action='store_true',
        help='Enable image corruption mCE eval at validation',
        default=True
    )

    parser.add_argument(
        '--eval-corruption-image-list',
        type=str,
        default='jpeg_compression',
        help='JSON-list or comma-separated list of corruption names (e.g. \'[\"gaussian_noise\",\"defocus_blur\"]\') or subset name like "noise". If None, defaults to ["gaussian_noise","defocus_blur","jpeg_compression"].'
    )

    parser.add_argument(
        '--eval-corruption-severities',
        type=str,
        default='[3]',
        help='JSON-list of severities (e.g. "[1,2,3,4,5]" or "[3]")'
    )

    parser.add_argument(
        '--eval-text-corruption',
        # action='store_true',
        help='Enable simple text-level corruption at validation',
        default=True
    )

    parser.add_argument(
        '--eval-text-corruption-severity',
        type=float,
        default=0.3,
        help='For text corruption: fraction of tokens to corrupt (0..1), default 0.3'
    )

    # ----------------- NEW ARGS FOR ABSOLUTE mCE -----------------
    parser.add_argument(
        '--eval-corruption-baseline',
        type=str,
        default='epoch0',
        help="Baseline mode for absolute mCE: 'epoch0' (default) or 'custom'. If 'custom', provide numeric baseline mAPs via --eval-corruption-baseline-mapi2t and --eval-corruption-baseline-mapt2i."
    )

    parser.add_argument(
        '--eval-corruption-baseline-mapi2t',
        type=float,
        default=None,
        help='Optional: explicit baseline clean mAP for i->t (float in 0..1). If set together with --eval-corruption-baseline-mapt2i, absolute mCE will use these values.'
    )
    parser.add_argument(
        '--eval-corruption-baseline-mapt2i',
        type=float,
        default=None,
        help='Optional: explicit baseline clean mAP for t->i (float in 0..1).'
    )

    args = parser.parse_args()

    args.method = 'IDML'

    return args

args = get_args()
sheet = xlrd.open_workbook('./utils/codetable.xlsx').sheet_by_index(0)
threshold = sheet.row(args.output_dim)[math.ceil(math.log(args.n_classes, 2))].value
