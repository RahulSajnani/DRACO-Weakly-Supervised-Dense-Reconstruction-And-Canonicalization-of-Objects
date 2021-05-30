"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
import os, hydra, logging, glob
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

# Training models
from trainers import DRACO_phase_1 as training_model_1
from trainers import DRACO_phase_2 as training_model_2


seed_everything(123)
log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config_DRACO.yaml")
def run(cfg):

    log.info(cfg.pretty())

    checkpoint_callback = pl.ModelCheckpoint(**cfg.callback.model_checkpoint.depth.args)
    model = training_model_1(hparams = cfg)

    trainer = Trainer(**cfg.trainer, callbacks = [checkpoint_callback])
    trainer.fit(model)


    checkpoint_file = glob.glob("./**.ckpt")[0]
    model_depth = training_model_1.load_from_checkpoint(checkpoint_file)
    
    ############# Phase 2 training the NOCS decoder
    checkpoint_callback_2 = pl.ModelCheckpoint(**cfg.callback.model_checkpoint.nocs.args)

    print("\n[Info] Starting Phase 2 training (NOCS decoder) using depth checkpoint from path: ", checkpoint_file)
    model_2 = training_model_2(hparams = cfg, depth_model = model_depth.model)
    trainer = Trainer(**cfg.trainer, callbacks = [checkpoint_callback_2])
    trainer.fit(model_2)

if __name__ == '__main__':

    run()
