from lightning.pytorch import LightningModule
from ViewDelta.model.model_feature_segmentor import TextConditionedDecoder
from ViewDelta.model.transformer_args import TransformerModelArgs
from kornia.losses import dice_loss, focal_loss
import torch
import os
import math
import random


def random_augment(image_a, image_b, labels):
    """
    Applies the same random augmentation to image_a, image_b, and labels.
    Args:
    image_a (torch.Tensor): Tensor of shape (B, tokens, embedding_dim).
    image_b (torch.Tensor): Tensor of shape (B, tokens, embedding_dim).
    labels (torch.Tensor): Tensor of shape (B, H, W).
    Returns:
    tuple: Augmented image_a, image_b, and labels.
    """
    # Handle extra dimensions
    original_shape = image_a.shape
    if len(original_shape) == 4:
        image_a = image_a.squeeze(1)
        image_b = image_b.squeeze(1)

    B, tokens, embedding_dim = image_a.shape

    # Extract cls tokens and remove them from image_a and image_b
    cls_a = image_a[:, -1:, :]
    cls_b = image_b[:, -1:, :]
    image_a_no_cls = image_a[:, :-1, :]
    image_b_no_cls = image_b[:, :-1, :]

    # Calculate token spatial dimensions
    token_count = image_a_no_cls.shape[1]
    token_H = int(token_count**0.5)
    token_W = token_H

    # Reshape image embeddings to match label dimensions
    image_a_reshaped = image_a_no_cls.reshape(
        B, token_H, token_W, embedding_dim
    ).permute(
        0, 3, 1, 2
    )  # B, embedding_dim, H, W

    image_b_reshaped = image_b_no_cls.reshape(
        B, token_H, token_W, embedding_dim
    ).permute(0, 3, 1, 2)

    # Store a list of applied augmentations for reproducibility if needed.
    applied_augmentations = []

    # Random flip up/down
    if random.random() > 0.3:
        image_a_reshaped = torch.flip(image_a_reshaped, dims=[2])
        image_b_reshaped = torch.flip(image_b_reshaped, dims=[2])
        labels = torch.flip(
            labels, dims=[1]
        )  # Always use dim 1 for height in 3D labels
        applied_augmentations.append("flipud")

    # Random flip left/right
    if random.random() > 0.3:
        image_a_reshaped = torch.flip(image_a_reshaped, dims=[3])
        image_b_reshaped = torch.flip(image_b_reshaped, dims=[3])
        labels = torch.flip(labels, dims=[2])  # Always use dim 2 for width in 3D labels
        applied_augmentations.append("fliplr")

    # Random rotation by 90 degrees (k=1, 2, or 3)
    k = random.randint(0, 3)
    if k > 0:
        image_a_reshaped = torch.rot90(image_a_reshaped, k=k, dims=(2, 3))
        image_b_reshaped = torch.rot90(image_b_reshaped, k=k, dims=(2, 3))
        labels = torch.rot90(labels, k=k, dims=(1, 2))  # Always use (1,2) for 3D labels
        applied_augmentations.append(f"rot90(k={k})")

    # Reshape image embeddings back to original token format
    image_a_augmented = image_a_reshaped.permute(0, 2, 3, 1).reshape(
        B, -1, embedding_dim
    )
    image_b_augmented = image_b_reshaped.permute(0, 2, 3, 1).reshape(
        B, -1, embedding_dim
    )

    # Concatenate cls tokens back
    image_a_augmented = torch.cat([image_a_augmented, cls_a], dim=1)
    image_b_augmented = torch.cat([image_b_augmented, cls_b], dim=1)

    # add back the empty dim if it was removed
    if len(original_shape) == 4:
        image_a_augmented = image_a_augmented.unsqueeze(1)
        image_b_augmented = image_b_augmented.unsqueeze(1)

    return (image_a_augmented, image_b_augmented, labels)

def get_incremented_dir_name(dir_name):
    if not os.path.exists(dir_name):
        return dir_name
    base_name = dir_name
    counter = 1
    while os.path.exists(dir_name):
        dir_name = f"{base_name}_{counter}"
        counter += 1
    return dir_name


# pytorch lightning wrapper over pytorch model of ViewDelta
class PLModelFeatureSegmentor(LightningModule):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.model = TextConditionedDecoder(args)
        self.model_args = args
        self.wandb_logger = self.logger
        self.random_offset = 0

        # log model hyperparameters in model_args
        self.save_hyperparameters(args.dict_convert())

        # if the output directory is not present, create it if it is present add a digit to the end
        # check if the directory is present
        self.model_args.output_dir = args.output_dir
        # create the directory
        os.makedirs(self.model_args.output_dir, exist_ok=True)
        self.train_losses = []  # Store the individual losses
        

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, preds, labels):
        loss = 20 * focal_loss(
            preds,
            labels,
            alpha=self.model_args.alpha,
            gamma=self.model_args.gamma,
            reduction="mean",
        ) + dice_loss(preds, labels, average=self.model_args.dice_average)

        # print(f"loss: {loss}")
        return loss

    def training_step(self, batch, batch_idx):
        image_a, image_b, text, text_embedding, label, label_name, dataset_name = (
            batch["image_a"],
            batch["image_b"],
            batch["text"],
            batch["text_embedding"],
            batch["label"],
            batch["label_name"],
            batch["dataset_name"],
        )

        preds = self.model(image_a, image_b, text_embedding)
        loss = self.loss_fn(preds, label)

        # logs the loss at fixed intervals
        self.train_losses.append(loss)
        if (batch_idx) % (self.model_args.log_freq) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log(
                f"avg_train_loss", avg_loss, on_step=True, prog_bar=True, sync_dist=True
            )

            self.train_losses = []  # Reset the losses after logging
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True, on_step=True, logger=True)
        if (batch_idx) % (1000) == 0:
            self.save_checkpoint(
                f"{self.model_args.output_dir}/{self.global_step}_checkpoint.pth"
            )
        return loss

    def lr_lambda_func(self, epoch):
        if epoch < self.model_args.warmup_epochs:
            # Linear warm-up
            return float(epoch + 1) / float(self.model_args.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.model_args.warmup_epochs) / float(
                max(1, self.model_args.epochs - self.model_args.warmup_epochs)
            )
            lr = 0.5 * (1.0 + math.cos(math.pi * progress))
            print(f"lr: {lr}")
            return lr

    def configure_optimizers(self):
        optimizer = self.model_args.optimizer(
            self.model.parameters(), lr=self.model_args.learning_rate, fused=True
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.lr_lambda_func
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Saved checkpoint to {filename}")
