import os
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import DCGANLikeModel
from ign import IdempotentNetwork


def main(args):
    # Set seed
    pl.seed_everything(args["seed"])

    # Load datas
    normalize = Lambda(lambda x: (x - 0.5) * 2)
    transform = Compose([ToTensor(), normalize])

    train_set = MNIST(root="mnist", train=True, download=True, transform=transform)
    val_set = MNIST(root="mnist", train=False, download=True, transform=transform)

    collate_fn = lambda samples: torch.stack([sample[0] for sample in samples])
    train_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args["num_workers"],
    )

    # Initialize model
    prior = torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28))
    net = DCGANLikeModel()
    model = IdempotentNetwork(prior, net, args["lr"])

    # Train model
    logger = WandbLogger(name="IGN", project="Papers Re-implementations")
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss", mode="min", dirpath="checkpoints", filename="best",
        )
    ]
    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="auto",
        max_epochs=args["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    # Loading the best model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        IdempotentNetwork.load_from_checkpoint("checkpoints/best.ckpt", prior=prior, model=net)
        .eval()
        .to(device)
    )

    # Generating images with the trained model
    os.makedirs("generated", exist_ok=True)

    unnormalize = Lambda(lambda x: (x / 2) + 0.5)
    transform = Compose([unnormalize, ToPILImage()])

    images = model.generate_n(10, device=device)
    for i, img in enumerate(images):
        transform(img).save(f"generated/{i+1}.png")

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    args = vars(parser.parse_args())

    print("\n\n", args, "\n\n")
    main(args)
