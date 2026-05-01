
from train.hash_train_RSH import Trainer
from utils import get_args
import os
import copy

def get_class_num(name):
    r = {"coco": 80, "25K": 24, "nuswide": 21, "IAPR": 291}[name]
    return r

if __name__ == "__main__":

    args = get_args()

    for ba in [128]:

        for noise in [0.8]:

                for dataset in ["25K"]:

                    print(f"processing dataset: {dataset}")

                    for hash_bit in [128]:

                        args = get_args()

                        args.save_dir = './result/'

                        args.noise_rate = noise

                        args.epochs = 101
                        args.tau1 = 9

                        args.gamma1 = 3


                        args.batch_size = ba
                        if dataset == "nuswide":
                            args.caption_file = "caption.txt"
                        else:
                            args.caption_file = "caption.mat"
                        args.dataset = dataset
                        args.n_classes = get_class_num(dataset)

                        args.n_bits = hash_bit
                        args.output_dim = hash_bit
                        args.save_dir = os.path.join(args.save_dir, args.dataset, str(args.output_dim), str(args.noise_rate))

                        Trainer(args)
