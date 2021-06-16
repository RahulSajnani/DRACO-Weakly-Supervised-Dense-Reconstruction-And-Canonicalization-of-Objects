import sys, os

from torch.utils import tensorboard
sys.path.append(os.path.abspath(os.path.join('./models')))
sys.path.append(os.path.abspath(os.path.join('./Data_Loaders')))
sys.path.append(os.path.abspath(os.path.join('./Loss_Functions')))

"""
This file defines the core research contribution
"""
import pytorch_lightning as pl
import models, argparse, tqdm, gc, torch, torchvision, logging
from argparse import ArgumentParser
import data_loader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
import smoothness_loss, geometric_loss, photometric_loss

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from hydra import utils
from helper_functions import *
from nocs_generator import *

log = logging.getLogger(__name__)


class DRACO_phase_1(pl.LightningModule):

    def __init__(self, hparams: DictConfig = None):
        super(DRACO_phase_1, self).__init__()

        self.hparams = hparams
        self.model = getattr(getattr(models, self.hparams.model.file), self.hparams.model.type)(**self.hparams.model.args)
        self.vgg_model = torchvision.models.vgg16(pretrained=True).eval().features[:9]

        self.BCELoss          = torch.nn.BCELoss(reduction="none")
        self.Photometric_loss = photometric_loss.Photometric_loss(vgg_model = self.vgg_model, **self.hparams.loss.photometric.args)
        self.Smoothness_loss  = smoothness_loss.Smoothness_loss()
        self.Geometric_loss   = geometric_loss.Geometric_loss()


        self.w_bce = self.hparams.loss.mask.weight
        self.w_photo = self.hparams.loss.photometric.weight
        self.w_smooth = self.hparams.loss.smoothness.weight
        self.w_geometric = self.hparams.loss.geometric.weight
        self.least_loss = 100

        self.first_val = True

    def forward(self, x):

        return self.model.forward(x)


    def configure_optimizers(self):

        optimizer1 = getattr(torch.optim, self.hparams.optimizer.type)(self.model.parameters(), **self.hparams.optimizer.args)
        scheduler1 = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.type)(optimizer1, **self.hparams.scheduler.args)

        return [optimizer1], [scheduler1]

    def train_dataloader(self):

        self.hparams.dataset.train.args['dataset_path'] = os.path.join(utils.get_original_cwd(),self.hparams.dataset.train.args['dataset_path'])
        train_data_set = getattr(data_loader, self.hparams.dataset.train.type)(**self.hparams.dataset.train.args, transform = getattr(data_loader, self.hparams.dataset.train.type).ToTensor())
        train_dataloader = DataLoader(train_data_set, **self.hparams.dataset.train.loader.args)

        return train_dataloader

    def val_dataloader(self):

        self.hparams.dataset.val.args['dataset_path'] = os.path.join(utils.get_original_cwd(),self.hparams.dataset.val.args['dataset_path'])
        val_data_set = getattr(data_loader, self.hparams.dataset.val.type)(**self.hparams.dataset.val.args, transform = getattr(data_loader, self.hparams.dataset.val.type).ToTensor())
        val_dataloader = DataLoader(val_data_set, **self.hparams.dataset.val.loader.args)

        return val_dataloader


    def forward_pass(self, batch, batch_idx):

        target_mask = batch['masks'][:, 0].float()
        batch['views'] = batch['views'].float()
        output = self.model(batch['views'][:, 0])

        # Forward Pass
        target_depths = [sigmoid_2_depth(output[0], scale_factor = self.hparams.utils.depth_scale)]

        for i in range(1, batch['num_views'][0]):
           target_depths.append(sigmoid_2_depth(self.model(batch['views'][:, i])[0], scale_factor = self.hparams.utils.depth_scale))

        target_depths = torch.cat(target_depths, dim=1)

        # Weigh foreground pixels high
        weight_matrix = torch.ones(target_mask.shape).type_as(target_mask)
        foreground_area_ratio = torch.sum(target_mask > 0.5) / (torch.sum(weight_matrix) + 1e-6)
        weight_matrix[target_mask > 0.5] = weight_matrix[target_mask > 0.5] * (1 / (foreground_area_ratio + 0.00000001))


        # Loss Computation
        bce_loss            = self.w_bce         * (self.BCELoss(output[1], target_mask) * weight_matrix).mean()
        photometric_loss    = self.w_photo       * self.Photometric_loss(batch, target_depths)
        smoothness_loss     = self.w_smooth      * self.Smoothness_loss(output[0],batch)


        loss = bce_loss + photometric_loss + smoothness_loss

        # geometric_loss = torch.tensor(0.0).type_as(bce_loss)

        #if self.current_epoch > 20:
        #   geometric_loss      = self.w_geometric   * self.Geometric_loss(batch, target_depths)
        #   loss += geometric_loss

        return {'loss': loss, 'bce_loss': bce_loss / (self.w_bce + 1e-6),
                'photometric_loss': photometric_loss / (self.w_photo + 1e-6),
                'smoothness_loss': smoothness_loss / (self.w_smooth + 1e-6),
                # 'geometric_loss': geometric_loss / (self.w_geometric + 1e-6)
                }

    def training_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log_loss_dict(loss_dictionary)

        return loss_dictionary["loss"]


    def validation_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log("val_loss", loss_dictionary["loss"], on_epoch=True, prog_bar=True, logger=True)
        self.log_loss_dict(loss_dictionary)

        return loss_dictionary["loss"]


    def log_loss_dict(self, loss_dictionary):

        for key in loss_dictionary:
            self.log(key, loss_dictionary[key], on_epoch=True, prog_bar=True, logger=True, on_step=True)



class DRACO_phase_2(pl.LightningModule):

    def __init__(self, hparams: DictConfig = None, depth_model = None, checkpoint_file_decoder=None):
        super(DRACO_phase_2, self).__init__()
        self.hparams = hparams

        ################## Model initialization

        self.model = getattr(getattr(models, self.hparams.model.file), self.hparams.model.type)(**self.hparams.model.args)
        if depth_model is None:
            self.model = getattr(getattr(models, self.hparams.model.file), self.hparams.model.type)(**self.hparams.model.args)
            print("[Warning]: No pretrained depth model is found.")
        else:
            self.model.load_state_dict(depth_model.state_dict())
            print("[INFO]: Using pretrained depth network and training NOCS")
        self.model.train()

        self.nocs_decoder = getattr(models, self.hparams.model.file).NOCS_decoder().train()
        if checkpoint_file_decoder is not None:
            self.nocs_decoder.load_state_dict(torch.load(os.path.abspath(checkpoint_file_decoder)))

        self.vgg_model = torchvision.models.vgg16(pretrained=True).eval().features[:9]
        #####################################################################################################

        ################## Loss functions
        self.BCELoss          = torch.nn.BCELoss()
        self.Photometric_loss = photometric_loss.Photometric_loss(vgg_model = self.vgg_model, **self.hparams.loss.photometric.args)
        self.Smoothness_loss  = smoothness_loss.Smoothness_loss()
        self.nocs_smoothness  = smoothness_loss.Smoothness_loss_nocs()
        self.nocs_photo       = photometric_loss.Photometric_loss_nocs(**self.hparams.loss.photometric.args)
        self.Geometric_loss   = geometric_loss.Geometric_loss()
        self.perceptual_loss  = photometric_loss.LossNetwork(self.vgg_model).eval()
        ######################################################################################################

        ################## Loss function weights
        self.w_bce = self.hparams.loss.mask.weight
        self.w_photo = self.hparams.loss.photometric.weight
        self.w_smooth = self.hparams.loss.smoothness.weight
        self.w_geometric = self.hparams.loss.geometric.weight
        ######################################################################################################

        self.least_loss = 100
        self.train_epoch_nocs = -1
        self.first_val = True


    def forward(self, x):

        return self.model.forward(x)


    def configure_optimizers(self):

        optimizer1 = getattr(torch.optim, self.hparams.optimizer.type)([{"params": self.nocs_decoder.parameters()},
                                                                        {"params": self.model.parameters(), "lr": 1e-6}], **self.hparams.optimizer.args)
        scheduler1 = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.type)(optimizer1, **self.hparams.scheduler.args)

        return [optimizer1], [scheduler1]

    def train_dataloader(self):

        self.hparams.dataset.train.args['dataset_path'] = os.path.join(utils.get_original_cwd(),self.hparams.dataset.train.args['dataset_path'])
        train_data_set = getattr(data_loader, self.hparams.dataset.train.type)(**self.hparams.dataset.train.args, transform = getattr(data_loader, self.hparams.dataset.train.type).ToTensor())
        train_dataloader = DataLoader(train_data_set, **self.hparams.dataset.train.loader.args)


        return train_dataloader

    def val_dataloader(self):

        self.hparams.dataset.val.args['dataset_path'] = os.path.join(utils.get_original_cwd(),self.hparams.dataset.val.args['dataset_path'])
        val_data_set = getattr(data_loader, self.hparams.dataset.val.type)(**self.hparams.dataset.val.args, transform = getattr(data_loader, self.hparams.dataset.val.type).ToTensor())
        val_dataloader = DataLoader(val_data_set, **self.hparams.dataset.val.loader.args)


        return val_dataloader

    def test_pass(self, batch):

        out = self.model.forward(batch,  encoder = True)
        depth = sigmoid_2_depth(out[0], scale_factor = self.hparams.utils.depth_scale)
        mask = out[1]
        return depth, mask, out[2]

    def forward_pass(self, batch, batch_idx):

        target_mask = batch['masks'][:, 0].float()
        batch['views'] = batch['views'].float()

        output = self.model(batch['views'][:, 0], encoder = True)


        target_depths = [sigmoid_2_depth(output[0], scale_factor = self.hparams.utils.depth_scale)]

        if self.current_epoch > self.train_epoch_nocs:
            target_nocs = [self.nocs_decoder(output[2])]

        for i in range(1, batch['num_views'][0]):
            outputs_view = self.model(batch['views'][:, i], encoder = True)
            target_depths.append(sigmoid_2_depth(outputs_view[0], scale_factor = self.hparams.utils.depth_scale))

            if self.current_epoch > self.train_epoch_nocs:
                target_nocs.append(self.nocs_decoder(outputs_view[2]))


        target_depths = torch.cat(target_depths, dim = 1)


        # Loss Computation
        bce_loss            = self.w_bce         * self.BCELoss(output[1], target_mask)
        photometric_loss    = self.w_photo       * self.Photometric_loss(batch, target_depths)
        smoothness_loss     = self.w_smooth      * self.Smoothness_loss(output[0], batch)

        loss = bce_loss + photometric_loss + smoothness_loss
        nocs_loss = torch.tensor(0.0).type_as(batch["intrinsics"])


        if self.current_epoch > self.train_epoch_nocs:

            target_nocs = torch.stack(target_nocs, dim = 1)
            nocs_generated = generate_nocs(batch, (target_depths.unsqueeze(2).detach() * batch["masks"].float())).detach()

            perceptual_loss_nocs = 0
            nocs_smoothness_loss = 0
            nocs_masked = target_nocs[:, 0] * batch["masks"][:, 0].float()

            nocs_smoothness_loss += self.nocs_smoothness(nocs_masked, batch)
            perceptual_loss_nocs += (self.perceptual_loss(nocs_generated[:, 0] * batch["masks"][:, 0].float()) - self.perceptual_loss(nocs_masked)).abs().mean()

            nocs_loss = torch.mean(torch.abs(nocs_generated.detach() - target_nocs))

            # Detaching target depths as we do not wish to back propagate through predicted depths
            batch["nocs"] = target_nocs
            nocs_photometric_loss = self.nocs_photo(batch, target_depths.detach())#
            nocs_loss = torch.sum(torch.abs(nocs_generated - target_nocs) * batch["masks"].float()) / torch.sum(batch["masks"] > 0.5)
            loss += 0.3 * nocs_smoothness_loss +  nocs_photometric_loss

            # If Umeyama alignment (rare case) fails then the transformed NOCS has inf values and so we do not penalize in such a case
            if not torch.isnan(nocs_generated).any():
                loss += nocs_loss +  2 * perceptual_loss_nocs


        return {'loss': loss,
                "nocs_loss": nocs_loss,
                'bce_loss': bce_loss / (self.w_bce + 1e-6),
                'photometric_loss': photometric_loss / (self.w_photo + 1e-6),
                'smoothness_loss': smoothness_loss / (self.w_smooth + 1e-6),
                # 'geometric_loss': geometric_loss / self.w_geometric,
                }


    def training_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log_loss_dict(loss_dictionary)
        return loss_dictionary["loss"]


    def validation_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log("val_loss", loss_dictionary["loss"], on_epoch=True, prog_bar=True, logger=True)
        self.log_loss_dict(loss_dictionary)

        return loss_dictionary["loss"]


    def log_loss_dict(self, loss_dictionary):

        for key in loss_dictionary:
            self.log(key, loss_dictionary[key], on_epoch=True, prog_bar=True, logger=True, on_step=True)


