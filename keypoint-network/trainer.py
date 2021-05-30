
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from Data_loaders import data_loader
from models import hourglass


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self, nstack, weighted = False):
        super().__init__()
        self.nstack = nstack
        self.weighted = weighted

    def loss_single(self, pred, ground_truth):

        weights = (ground_truth > 0.1) * 81 + 1

        criterion = torch.nn.BCEWithLogitsLoss()

        if self.weighted:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
            #print("check")
        #    l = ((pred - ground_truth)**2) * weights

        #else:
        #    l = ((pred - ground_truth)**2)
        l = criterion(pred, ground_truth)
        #l = l.mean(dim=3).mean(dim=2).mean(dim=1)   #[4, 16, 64, 64] -> [4]
        #print(l)
        return l    # size = batch_size

    def forward(self, combined_heatmap_preds, heatmaps_gt):

        combined_loss = []

        for i in range(self.nstack):
            combined_loss.append(self.loss_single(combined_heatmap_preds[: ,i], heatmaps_gt))

        #print(combined_loss)
        combined_loss = torch.stack(combined_loss, dim=0)
        mean_loss = torch.mean(combined_loss)

        return mean_loss

class HG_trainer(LightningModule):

    def __init__(self, batch_size, dataset_path, nstack = 3, nclasses = 22, nblocks = 4, weighted = False, **kwargs):
        super(HG_trainer, self).__init__()
        self.dataset_path = dataset_path
        self.hparams.batch_size = batch_size
        self.hparams.num_workers = 4
        self.nstack = nstack
        self.nblock = nblocks
        self.weighted = weighted
        self.network = hourglass.hg(num_classes = nclasses, num_stacks = nstack, num_blocks = nblocks)
        self.calc_loss = HeatmapLoss(nstack, weighted)
        self.least_loss = 100
        self.least_loss_val = 100

    def forward(self, imgs):

        all_pred_heatmaps = self.network(imgs)

        return torch.stack(all_pred_heatmaps, dim=1)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass

        batch_imgs, heatmaps_gt = batch["views"], batch["heatmaps"]
        combined_heatmap_preds = self(batch_imgs)

        train_loss = self.calc_loss(combined_heatmap_preds, heatmaps_gt)
        epoch_trainer_logger = {"train_loss": train_loss}

        return {"loss": train_loss, "train_epoch_logger": epoch_trainer_logger, "log": epoch_trainer_logger}


    def validation_step(self, batch, batch_idx):
        """
        Called every batch
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        batch_imgs, heatmaps_gt = batch["views"], batch["heatmaps"]
        combined_heatmap_preds = self(batch_imgs)

        val_loss = self.calc_loss(combined_heatmap_preds, heatmaps_gt)
        log_tb = {"val_loss": val_loss}

        return {"val_loss": val_loss, "log": log_tb, "val_epoch_logger": log_tb}

    '''
    def test_step(self, batch, batch_idx):

        batch_imgs, heatmaps_gt = batch["views"], batch["heatmaps"]
        combined_heatmap_preds = self(batch_imgs)

        test_loss = self.calc_loss(combined_heatmap_preds, heatmaps_gt)
        tensorboard_logs = {'test_loss': test_loss}

        return {'test_loss': test_loss, 'test_log': tensorboard_logs}

    '''

    def training_epoch_end(self, outputs):
        '''
        Logging all losses at the end of training epoch
        '''

        epoch_train_loss = torch.stack([x['train_epoch_logger']['train_loss'] for x in outputs]).mean()

        print("\nTrain loss:", epoch_train_loss)

        return {"train_avg_loss": epoch_train_loss}

    def validation_epoch_end(self, outputs):
        '''
        Logging all losses at the end of training epoch
        '''

        epoch_val_loss = torch.stack([x['val_epoch_logger']['val_loss'] for x in outputs]).mean()

        print("\nValidation loss:", epoch_val_loss)

        if self.least_loss_val > epoch_val_loss:
            self.least_loss_val = epoch_val_loss
            self.save_model()

        pbar = {"val_loss": epoch_val_loss}
        return {"val_avg_loss": epoch_val_loss, "progress_bar": pbar}

    def save_model(self):

        '''
        Custom save model function for SLURM
        '''
        if self.weighted:
            save_file_name = f"./hg_rms_large{self.nstack}_{self.nblock}_change_{self.weighted*1}.ckpt"
        else:
            save_file_name = f"./hg_rms_no_weights_{self.nstack}_{self.nblock}large_change_{self.weighted*1}.ckpt"

        torch.save(self.network.state_dict(), save_file_name)
        print(f"Saved model in location  {save_file_name}")

    '''
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}
    '''

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.network.parameters(), lr=2.5e-4)
        #optimizer = torch.optim.Adam(self.network.parameters(), lr=2.5e-5, weight_decay = 0.004, betas= (0.009, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, verbose = True)
        return [optimizer], [scheduler]


    def setup(self, stage):

        self.train_set = data_loader.Keypoint_dataset(self.dataset_path, train = 1)
        self.val_set = data_loader.Keypoint_dataset(self.dataset_path, train = 2)
        self.test_set = data_loader.Keypoint_dataset(self.dataset_path, train = 0)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle = False)

