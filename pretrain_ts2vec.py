"""
TS2Vec Pre-training Script

Pre-train the TS2Vec encoder with contrastive learning before end-to-end training.
This can improve the quality of vector representations and potentially speed up convergence.

Usage:
    python pretrain_ts2vec.py --data_path ETTm1 --epochs 50 --batch_size 256
"""

import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.TS2Vec import TS2VecEncoder
from utils.contrastive_loss import TS2VecLoss
from utils.augmentation import RandomAugmentor


def parse_args():
    parser = argparse.ArgumentParser(description='TS2Vec Pre-training')

    # Data
    parser.add_argument("--data_path", type=str, default="ETTm1", help="dataset name")
    parser.add_argument("--seq_len", type=int, default=96, help="sequence length")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of variables")

    # Model
    parser.add_argument("--d_vector", type=int, default=320, help="output dimension")
    parser.add_argument("--hidden_dims", type=int, default=64, help="hidden dimension")
    parser.add_argument("--depth", type=int, default=10, help="encoder depth")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")

    # Contrastive learning
    parser.add_argument("--temperature", type=float, default=0.05, help="contrastive temperature")
    parser.add_argument("--n_augmentations", type=int, default=2, help="number of augmentations")
    parser.add_argument("--aug_prob", type=float, default=0.6, help="augmentation probability")

    # Misc
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--num_workers", type=int, default=10, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--save_path", type=str, default="./checkpoints/ts2vec_pretrain/",
                        help="save directory")

    return parser.parse_args()


def load_data(args):
    """Load dataset for pre-training"""
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)

    # Only need training data for pre-training
    train_set = data_class(
        flag='train',
        scale=True,
        size=[args.seq_len, 0, 96],  # pred_len doesn't matter for pre-training
        data_path=args.data_path
    )

    val_set = data_class(
        flag='val',
        scale=True,
        size=[args.seq_len, 0, 96],
        data_path=args.data_path
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )

    return train_loader, val_loader


def seed_it(seed):
    """Set random seeds"""
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_it(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading {args.data_path} dataset...")
    train_loader, val_loader = load_data(args)
    print(f"Training samples: {len(train_loader) * args.batch_size}")
    print(f"Validation samples: {len(val_loader) * args.batch_size}")

    # Initialize model
    print("\nInitializing TS2Vec encoder...")
    encoder = TS2VecEncoder(
        input_dims=args.num_nodes,
        output_dims=args.d_vector,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        mask_mode='binomial'
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and loss
    optimizer = optim.Adam(
        encoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    contrastive_loss = TS2VecLoss(
        alpha=0.5, beta=0.5, temperature=args.temperature
    )

    # Data augmentation
    augmentor = RandomAugmentor(
        augmentation_list=['jitter', 'scaling', 'permutation', 'magnitude_warp'],
        n_augmentations=args.n_augmentations,
        augmentation_prob=args.aug_prob
    )

    # Create save directory
    save_path = os.path.join(args.save_path, args.data_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("\n" + "="*60)
    print("Starting TS2Vec Pre-training")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print("="*60 + "\n")

    # Training
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        # Training
        encoder.train()
        epoch_losses = []
        hierarchical_losses = []
        temporal_losses = []

        t1 = time.time()

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
            x = x.to(device).float()  # [B, L, N]

            # Generate two augmented views
            x_aug1 = augmentor(x)
            x_aug2 = augmentor(x)

            # Get representations
            repr1 = encoder(x_aug1)  # [B, C, L]
            repr2 = encoder(x_aug2)  # [B, C, L]

            # Compute contrastive loss
            loss, loss_dict = contrastive_loss(repr1, repr2)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            hierarchical_losses.append(loss_dict['hierarchical'])
            temporal_losses.append(loss_dict['temporal'])

            if (iter + 1) % 50 == 0:
                print(f"  Batch {iter+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Hier: {loss_dict['hierarchical']:.4f} | "
                      f"Temp: {loss_dict['temporal']:.4f}")

        t2 = time.time()

        avg_train_loss = np.mean(epoch_losses)
        avg_hier_loss = np.mean(hierarchical_losses)
        avg_temp_loss = np.mean(temporal_losses)
        train_losses.append(avg_train_loss)

        # Validation
        encoder.eval()
        val_epoch_losses = []

        with torch.no_grad():
            for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(val_loader):
                x = x.to(device).float()

                # Two augmented views
                x_aug1 = augmentor(x)
                x_aug2 = augmentor(x)

                # Representations
                repr1 = encoder(x_aug1)
                repr2 = encoder(x_aug2)

                # Loss
                loss, _ = contrastive_loss(repr1, repr2)
                val_epoch_losses.append(loss.item())

        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)

        # Logging
        print(f"\nEpoch {epoch}/{args.epochs} | Time: {t2-t1:.2f}s")
        print(f"  Train - Total: {avg_train_loss:.4f} | "
              f"Hierarchical: {avg_hier_loss:.4f} | "
              f"Temporal: {avg_temp_loss:.4f}")
        print(f"  Valid - Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                encoder.state_dict(),
                os.path.join(save_path, 'best_ts2vec_encoder.pth')
            )
            print(f"  >>> Best model saved! (epoch {epoch})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                },
                os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
            )

        scheduler.step()
        print()

    # Final save
    torch.save(
        encoder.state_dict(),
        os.path.join(save_path, 'final_ts2vec_encoder.pth')
    )

    print("="*60)
    print("Pre-training Completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_path}")
    print("="*60)

    # Plot training curve (optional)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('TS2Vec Pre-training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'training_curve.png'))
        print(f"Training curve saved to {save_path}/training_curve.png")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"\nTotal time: {(t2-t1)/60:.2f} minutes")
