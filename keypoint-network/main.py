from pytorch_lightning import Trainer
from trainer import HG_trainer
from pytorch_lightning.callbacks import EarlyStopping
import argparse

if __name__ == "__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--nstack", default = 3, type = int)
    parser.add_argument("--nblock", default = 2, type = int)
    parser.add_argument("--batch_size", default = 20, type = int)
    parser.add_argument("--dataset", help="Dataset path", required=True)
    parser.add_argument("--weighted", default=True, type=int, choices=[0, 1])

    args = parser.parse_args()
    #######################################################################################

    print(args)
    net = HG_trainer(batch_size = args.batch_size, dataset_path = args.dataset, nstack = args.nstack, nclasses = 22, nblocks = args.nblock, weighted = args.weighted)

    #early_stopping = EarlyStopping('val_loss')
    trainer = Trainer(gpus=-1, distributed_backend='dp')
    trainer.fit(net)
