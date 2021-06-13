"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
import os, hydra, logging, glob
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# Training models
from trainers import DRACO_phase_1 as training_model_1
from trainers import DRACO_phase_2 as training_model_2


seed_everything(123)
log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config_DRACO.yaml")
def run(cfg):

    log.info(cfg)
    print(os.getcwd())
    checkpoint_callback = ModelCheckpoint(**cfg.callback.model_checkpoint.depth.args)
    model = training_model_1(hparams = cfg)

    trainer = Trainer(**cfg.trainer, callbacks = [checkpoint_callback])
    trainer.fit(model)

    print(os.getcwd())
    #checkpoint_file = glob.glob("./checkpoints/**.ckpt")[0]
    
    #checkpoint_file = glob.glob("/home/rahulsajnani/research/DRACO-Weakly-Supervised-Dense-Reconstruction-And-Canonicalization-of-Objects/DRACO/outputs/2021-06-12/02-55-56/checkpoints/*.ckpt")[0]
    model = training_model_1.load_from_checkpoint(checkpoint_path = checkpoint_file)

    ############# Phase 2 training the NOCS decoder
    checkpoint_callback_2 = ModelCheckpoint(**cfg.callback.model_checkpoint.nocs.args)

    print("\n[Info] Starting Phase 2 training (NOCS decoder) using depth checkpoint from path: ", checkpoint_file)
    model_2 = training_model_2(hparams = cfg, depth_model = model.model)
    trainer = Trainer(**cfg.trainer, callbacks = [checkpoint_callback_2])
    trainer.fit(model_2)

if __name__ == '__main__':

    run()
