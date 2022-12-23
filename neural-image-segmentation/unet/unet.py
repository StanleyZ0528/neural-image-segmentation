"""
Based on lightning-bolts models
"""
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from typing import Optional, Sequence, Union

class UNet_model(nn.Module):
    """
    Based on:
    https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/vision/unet.py
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm (optional) => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Taken from https://github.com/justusschock/dl-utils/blob/master/dlutils/losses/soft_dice.py
class SoftDiceLoss(torch.nn.Module):
    """Soft Dice Loss"""
    def __init__(self, square_nom: bool = False,
                 square_denom: bool = False,
                 weight: Optional[Union[Sequence, torch.Tensor]] = None,
                 smooth: float = 1.):
        """
        Args:
            square_nom: whether to square the nominator
            square_denom: whether to square the denominator
            weight: additional weighting of individual classes
            smooth: smoothing for nominator and denominator

        """
        super().__init__()
        self.square_nom = square_nom
        self.square_denom = square_denom

        self.smooth = smooth

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)

            self.register_buffer("weight", weight)
        else:
            self.weight = None
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Computes SoftDice Loss

        Args:
            predictions: the predictions obtained by the network
            targets: the targets (ground truth) for the :attr:`predictions`

        Returns:
            torch.Tensor: the computed loss value
        """
        # number of classes for onehot
        n_classes = predictions.shape[1]
        with torch.no_grad():
            targets_onehot = F.one_hot(
                targets.unsqueeze(1), num_classes=n_classes)
        # sum over spatial dimensions
        dims = tuple(range(2, predictions.dim()))

        # compute nominator
        nom = torch.sum(predictions * targets.float(), dim=dims)
        nom = 2 * nom + self.smooth

        # compute denominator
        denom = torch.sum(predictions ** 2 + targets ** 2, dim=dims) + self.smooth

        # compute loss
        frac = nom / denom

        # apply weight for individual classesproperly
        if self.weight is not None:
            frac = self.weight * frac

        # average over classes
        frac = 1 - torch.mean(frac, dim=1)

        return frac


class UNet(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        num_classes: int = 4,
        num_layers: int = 5,
        input_channels: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        loss_weight: Optional[Sequence[float]] = None,
        ignore_index: Optional[int] = None,
    ):
        """Based on:
        https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/vision/segmentation.py
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.features_start = features_start
        self.ignore_index = ignore_index
        self.bilinear = bilinear
        self.lr = lr

        if loss_weight:
            self.register_buffer("loss_weight", torch.tensor(loss_weight))

        self.net = UNet_model(
            num_classes=self.num_classes,
            input_channels=self.input_channels,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )
        self.dice_loss = SoftDiceLoss(loss_weight)

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_bar = y_hat.argmax(dim=1)
        dice_loss = self.dice_loss(y_bar, y)
        ce_loss = F.cross_entropy(y_hat, y, weight=self.loss_weight, ignore_index=self.ignore_index)
        loss = (ce_loss + dice_loss) / 2
        
        log_dict = {"train_loss": loss.mean(), "ce_loss": ce_loss, "dice_loss": dice_loss}
        return {"loss": loss.mean(), "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_bar = y_hat.argmax(dim=1)
        ce_loss_val = F.cross_entropy(y_hat, y, weight=self.loss_weight, ignore_index=self.ignore_index)
        dice_loss_val = self.dice_loss(y_bar, y)
        loss_val = (dice_loss_val + ce_loss_val) / 2
        
        
        fg_mask = (y==1).logical_or(y==2)
        ne_mask = (y==2)
        val_acc    = torch.sum(y_bar==y).item()/(torch.numel(y))
        if torch.sum(fg_mask).item() > 0:
            val_fg_acc = torch.sum(y_bar[fg_mask]==y[fg_mask]).item() / torch.sum(fg_mask).item()
        else:
            val_fg_acc = torch.tensor(0.)
        if torch.sum(ne_mask).item() > 0:
            val_ne_acc = torch.sum(y_bar[ne_mask]==y[ne_mask]).item() / torch.sum(ne_mask).item()
        else:
            val_ne_acc = torch.tensor(0.)

        metric_dict= {"val_loss":       loss_val.mean(),
                      "val_acc":        val_acc,
                      "val_fg_acc":     val_fg_acc,
                      "val_ne_acc":     val_ne_acc
                     }
        self.log_dict(metric_dict, prog_bar=True, logger=True)
        return metric_dict
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_bar = y_hat.argmax(dim=1)
        ce_loss_test = F.cross_entropy(y_hat, y, weight=self.loss_weight, ignore_index=self.ignore_index)
        dice_loss_test = self.dice_loss(y_bar, y)
        loss_test = (dice_loss_test + ce_loss_test) / 2
        
        
        fg_mask = (y==1).logical_or(y==2)
        ne_mask = (y==2)
        test_acc    = torch.sum(y_bar==y).item()/(torch.numel(y))
        if torch.sum(fg_mask).item() > 0:
            test_fg_acc = torch.sum(y_bar[fg_mask]==y[fg_mask]).item() / torch.sum(fg_mask).item()
        else:
            test_fg_acc = torch.tensor(0.)
        if torch.sum(ne_mask).item() > 0:
            test_ne_acc = torch.sum(y_bar[ne_mask]==y[ne_mask]).item() / torch.sum(ne_mask).item()
        else:
            test_ne_acc = torch.tensor(0.)

        metric_dict= {"test_loss":       loss_test.mean(),
                      "test_acc":        test_acc,
                      "test_fg_acc":     test_fg_acc,
                      "test_ne_acc":     test_ne_acc
                      }
        self.log_dict(metric_dict, prog_bar=True, logger=True)
        return metric_dict

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"avg_val_loss": loss_val}
        return {"log": log_dict, "avg_val_loss": log_dict["avg_val_loss"], "progress_bar": log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]