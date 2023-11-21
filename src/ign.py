from copy import deepcopy

import torch
from torch.optim import Adam, SGD
from torch.nn import L1Loss
import pytorch_lightning as pl


class IdempotentNetwork(pl.LightningModule):
    def __init__(self, prior, model, lr=1e-4, criterion=L1Loss(), lrec_w=20.0, lidem_w=20.0, ltight_w=2.5):
        super(IdempotentNetwork, self).__init__()
        self.prior = prior
        self.model = model
        self.lr = lr
        self.model_copy = deepcopy(model)
        self.criterion = criterion
        self.lrec_w = lrec_w
        self.lidem_w = lidem_w
        self.ltight_w = ltight_w
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optim_copy = SGD(self.model_copy.parameters(), lr=self.lr)
        return optim, optim_copy

    def get_losses(self, x):
        # Updating copy
        self.model_copy.load_state_dict(self.model.state_dict())
        
        # Prior samples
        z = self.prior.sample_n(x.shape[0]).to(x.device)
        
        # Running models
        fx = self(x)
        fz = self(z)
        f_z = fz.detach()
        ff_z = self(f_z)
        f_fz = self.model_copy(fz)

        # Computing losses
        l_rec = self.lrec_w * self.criterion(fx, x)           # Only optimize model
        l_idem = self.lidem_w * self.criterion(f_fz, fz)      # Only optimize model (TODO: FIX --> copy is getting gradients here)
        l_tight = - self.ltight_w * self.criterion(ff_z, f_z) # Only optimize model_copy
        
        return l_rec, l_idem, l_tight

    def training_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        
        loss = l_rec + l_idem + l_tight
        
        optim, optim_copy = self.optimizers()
        
        optim.zero_grad()
        optim_copy.zero_grad()
        self.manual_backward(loss)
        optim.step()
        optim_copy.step()
        
        
        self.log_dict(
            {
                "train/loss_rec": l_rec,
                "train/loss_idem": l_idem,
                "train/loss_tight": l_tight,
                "train/loss": l_rec + l_idem + l_tight,
            },
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        self.log_dict(
            {
                "val/loss_rec": l_rec,
                "val/loss_idem": l_idem,
                "val/loss_tight": l_tight,
                "val/loss": l_rec + l_idem + l_tight,
            },
            sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        self.log_dict(
            {
                "test/loss_rec": l_rec,
                "test/loss_idem": l_idem,
                "test/loss_tight": l_tight,
                "test/loss": l_rec + l_idem + l_tight,
            },
            sync_dist=True
        )
        
    def generate_n(self, n, device=None):
        z = self.prior.sample_n(n)
        
        if device is not None:
            z = z.to(device)
        
        return self(z)
