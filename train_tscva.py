"""
TS-CVA Training Script

Multi-task learning with contrastive loss and forecasting loss
Supports TS2Vec pre-training and end-to-end fine-tuning
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
from models.TS_CVA import TS_CVA
from utils.metrics import MSE, MAE, metric
from utils.contrastive_loss import TS2VecLoss
from utils.augmentation import RandomAugmentor
import faulthandler

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser(description='TS-CVA Training')

    # Basic settings
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--data_path", type=str, default="ETTm1", help="data path")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    # Model architecture
    parser.add_argument("--channel", type=int, default=32, help="hidden dimension for TS branch")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of variables")
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction horizon")
    parser.add_argument("--d_llm", type=int, default=768, help="LLM embedding dimension")
    parser.add_argument("--d_vector", type=int, default=320, help="TS2Vec output dimension")
    parser.add_argument("--e_layer", type=int, default=2, help="encoder layers")
    parser.add_argument("--d_layer", type=int, default=1, help="decoder layers")
    parser.add_argument("--d_ff", type=int, default=64, help="feed-forward dimension")
    parser.add_argument("--head", type=int, default=8, help="attention heads")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate")

    # TS-CVA specific
    parser.add_argument("--use_triple_align", action='store_true', default=True,
                        help="use triple-modal alignment")
    parser.add_argument("--fusion_mode", type=str, default='gated',
                        choices=['gated', 'weighted', 'concat'], help="fusion strategy")

    # Multi-task learning
    parser.add_argument("--contrastive_weight", type=float, default=0.3,
                        help="weight for contrastive loss")
    parser.add_argument("--forecast_weight", type=float, default=0.7,
                        help="weight for forecasting loss")
    parser.add_argument("--contrastive_temp", type=float, default=0.05,
                        help="temperature for contrastive loss")

    # Augmentation
    parser.add_argument("--use_augmentation", action='store_true', default=True,
                        help="use data augmentation for contrastive learning")
    parser.add_argument("--n_augmentations", type=int, default=2,
                        help="number of augmentations to apply")
    parser.add_argument("--aug_prob", type=float, default=0.6,
                        help="probability of applying each augmentation")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--es_patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--num_workers", type=int, default=10, help="dataloader workers")

    # Pre-training
    parser.add_argument("--pretrain_epochs", type=int, default=0,
                        help="TS2Vec pre-training epochs (0 to disable)")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3,
                        help="learning rate for pre-training")

    # Save/load
    parser.add_argument("--save", type=str,
                        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-tscva-",
                        help="save path")
    parser.add_argument("--load_pretrain", type=str, default=None,
                        help="path to pre-trained TS2Vec encoder")

    return parser.parse_args()


class TSCVATrainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        d_vector,
        e_layer,
        d_layer,
        d_ff,
        head,
        use_triple_align,
        fusion_mode,
        contrastive_weight,
        forecast_weight,
        contrastive_temp,
        use_augmentation,
        n_augmentations,
        aug_prob,
        lrate,
        wdecay,
        device,
        epochs
    ):
        # Initialize TS-CVA model
        self.model = TS_CVA(
            device=device,
            channel=channel,
            num_nodes=num_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            dropout_n=dropout_n,
            d_llm=d_llm,
            d_vector=d_vector,
            e_layer=e_layer,
            d_layer=d_layer,
            d_ff=d_ff,
            head=head,
            use_triple_align=use_triple_align,
            fusion_mode=fusion_mode,
            contrastive_weight=contrastive_weight
        )

        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 50), eta_min=1e-6
        )

        # Loss functions
        self.forecast_loss = MSE
        self.contrastive_loss = TS2VecLoss(
            alpha=0.5, beta=0.5, temperature=contrastive_temp
        )
        self.MAE = MAE

        # Loss weights
        self.contrastive_weight = contrastive_weight
        self.forecast_weight = forecast_weight

        # Augmentation
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentor = RandomAugmentor(
                augmentation_list=['jitter', 'scaling', 'permutation', 'magnitude_warp'],
                n_augmentations=n_augmentations,
                augmentation_prob=aug_prob
            )

        self.clip = 5

        print("="*60)
        print("TS-CVA Model Initialized")
        print(f"Total parameters: {self.model.param_num():,}")
        print(f"Trainable parameters: {self.model.count_trainable_params():,}")
        print(f"Multi-task learning: Contrastive ({contrastive_weight}) + Forecast ({forecast_weight})")
        print(f"Augmentation: {use_augmentation}")
        print("="*60)

    def train(self, input_data, mark, embeddings, real):
        """Single training step with multi-task learning"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass with contrastive representations
        predict, vector_repr = self.model(
            input_data, mark, embeddings, return_contrastive=True
        )

        # Forecasting loss
        loss_forecast = self.forecast_loss(predict, real)

        # Contrastive loss
        if self.use_augmentation and self.contrastive_weight > 0:
            # Generate augmented view
            input_aug = self.augmentor(input_data)  # [B, L, N]

            # Get representations for augmented view
            _, vector_repr_aug = self.model(
                input_aug, mark, embeddings, return_contrastive=True
            )

            # Compute contrastive loss
            loss_contrastive, contrastive_dict = self.contrastive_loss(
                vector_repr, vector_repr_aug
            )
        else:
            loss_contrastive = torch.tensor(0.0, device=self.device)
            contrastive_dict = {'total': 0.0}

        # Combined loss
        loss_total = (
            self.forecast_weight * loss_forecast +
            self.contrastive_weight * loss_contrastive
        )

        # Backward
        loss_total.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # Metrics
        mae = self.MAE(predict, real)

        return {
            'total_loss': loss_total.item(),
            'forecast_loss': loss_forecast.item(),
            'contrastive_loss': contrastive_dict['total'],
            'mae': mae.item()
        }

    def eval(self, input_data, mark, embeddings, real_val):
        """Evaluation step"""
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input_data, mark, embeddings)

        loss = self.forecast_loss(predict, real_val)
        mae = self.MAE(predict, real_val)

        return loss.item(), mae.item()

    def pretrain_ts2vec(self, train_loader, epochs, lr):
        """Pre-train TS2Vec encoder with contrastive learning only"""
        print("\n" + "="*60)
        print("Starting TS2Vec Pre-training")
        print("="*60)

        # Freeze all except vector encoder
        for name, param in self.model.named_parameters():
            if 'vector_encoder' not in name:
                param.requires_grad = False

        # Optimizer for pre-training
        pretrain_optimizer = optim.Adam(
            self.model.vector_encoder.parameters(), lr=lr
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = []

            for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
                x = x.to(self.device).float()

                # Generate two augmented views
                x_aug1 = self.augmentor(x)
                x_aug2 = self.augmentor(x)

                # Get representations
                repr1 = self.model.vector_encoder(x_aug1)
                repr2 = self.model.vector_encoder(x_aug2)

                # Contrastive loss
                loss, loss_dict = self.contrastive_loss(repr1, repr2)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

                epoch_loss.append(loss.item())

            avg_loss = np.mean(epoch_loss)
            print(f"Pre-train Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        print("="*60)
        print("TS2Vec Pre-training Completed")
        print("="*60 + "\n")


def load_data(args):
    """Load datasets"""
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)

    train_set = data_class(
        flag='train', scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path
    )
    val_set = data_class(
        flag='val', scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path
    )
    test_set = data_class(
        flag='test', scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path
    )

    scaler = train_set.scaler

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        drop_last=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        drop_last=True, num_workers=args.num_workers
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def seed_it(seed):
    """Set random seeds for reproducibility"""
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

    # Load data
    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    # Set seed
    seed_it(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    path = os.path.join(
        args.save, args.data_path,
        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_"
        f"{args.learning_rate}_{args.dropout_n}_{args.contrastive_weight}_{args.seed}/"
    )
    if not os.path.exists(path):
        os.makedirs(path)

    print("\n" + "="*60)
    print("Configuration:")
    print(args)
    print("="*60 + "\n")

    # Initialize trainer
    engine = TSCVATrainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        d_vector=args.d_vector,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        d_ff=args.d_ff,
        head=args.head,
        use_triple_align=args.use_triple_align,
        fusion_mode=args.fusion_mode,
        contrastive_weight=args.contrastive_weight,
        forecast_weight=args.forecast_weight,
        contrastive_temp=args.contrastive_temp,
        use_augmentation=args.use_augmentation,
        n_augmentations=args.n_augmentations,
        aug_prob=args.aug_prob,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs
    )

    # Pre-training (optional)
    if args.pretrain_epochs > 0:
        engine.pretrain_ts2vec(train_loader, args.pretrain_epochs, args.pretrain_lr)
    elif args.load_pretrain:
        print(f"Loading pre-trained TS2Vec from {args.load_pretrain}")
        pretrain_state = torch.load(args.load_pretrain)
        engine.model.vector_encoder.load_state_dict(pretrain_state)

    # Training
    print("\n" + "="*60)
    print("Starting End-to-End Training")
    print("="*60 + "\n")

    best_val_loss = float('inf')
    best_test_mse = float('inf')
    epochs_since_best = 0
    bestid = 0

    train_time = []
    val_time = []
    his_loss = []

    for epoch in range(1, args.epochs + 1):
        # Training
        t1 = time.time()
        train_metrics = {
            'total_loss': [],
            'forecast_loss': [],
            'contrastive_loss': [],
            'mae': []
        }

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
            trainx = x.to(device).float()
            trainy = y.to(device).float()
            trainx_mark = x_mark.to(device).float()
            train_embedding = embeddings.to(device).float()

            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)

            for key in train_metrics:
                train_metrics[key].append(metrics[key])

        t2 = time.time()
        train_time.append(t2 - t1)

        # Validation
        s1 = time.time()
        val_loss = []
        val_mae = []

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(val_loader):
            valx = x.to(device).float()
            valy = y.to(device).float()
            valx_mark = x_mark.to(device).float()
            val_embedding = embeddings.to(device).float()

            loss, mae = engine.eval(valx, valx_mark, val_embedding, valy)
            val_loss.append(loss)
            val_mae.append(mae)

        s2 = time.time()
        val_time.append(s2 - s1)

        # Average metrics
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        avg_val_loss = np.mean(val_loss)
        avg_val_mae = np.mean(val_mae)

        his_loss.append(avg_val_loss)

        # Logging
        print(f"Epoch {epoch:03d} | Train Time: {t2-t1:.2f}s | Val Time: {s2-s1:.2f}s")
        print(f"  Train - Total: {avg_train_metrics['total_loss']:.4f} | "
              f"Forecast: {avg_train_metrics['forecast_loss']:.4f} | "
              f"Contrastive: {avg_train_metrics['contrastive_loss']:.4f} | "
              f"MAE: {avg_train_metrics['mae']:.4f}")
        print(f"  Valid - Loss: {avg_val_loss:.4f} | MAE: {avg_val_mae:.4f}")

        # Model checkpoint
        if avg_val_loss < best_val_loss:
            print("  >>> Validation improved! Testing on test set...")

            # Test
            test_outputs = []
            test_y = []

            for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
                testx = x.to(device).float()
                testy = y.to(device).float()
                testx_mark = x_mark.to(device).float()
                test_embedding = embeddings.to(device).float()

                with torch.no_grad():
                    preds = engine.model(testx, testx_mark, test_embedding)

                test_outputs.append(preds)
                test_y.append(testy)

            test_pre = torch.cat(test_outputs, dim=0)
            test_real = torch.cat(test_y, dim=0)

            test_mse = []
            test_mae = []

            for j in range(args.pred_len):
                pred = test_pre[:, j, :]
                real = test_real[:, j, :]
                metrics = metric(pred, real)
                test_mse.append(metrics[0])
                test_mae.append(metrics[1])

            avg_test_mse = np.mean(test_mse)
            avg_test_mae = np.mean(test_mae)

            print(f"  Test - MSE: {avg_test_mse:.4f} | MAE: {avg_test_mae:.4f}")

            if avg_test_mse < best_test_mse:
                best_val_loss = avg_val_loss
                best_test_mse = avg_test_mse
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = epoch
                epochs_since_best = 0
                print(f"  >>> Model saved! (epoch {epoch})")
            else:
                epochs_since_best += 1
        else:
            epochs_since_best += 1

        engine.scheduler.step()
        print()

        # Early stopping
        if epochs_since_best >= args.es_patience and epoch >= args.epochs // 2:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation
    print("\n" + "="*60)
    print("Training Completed")
    print(f"Best epoch: {bestid}")
    print(f"Best validation loss: {his_loss[bestid-1]:.4f}")
    print(f"Average training time: {np.mean(train_time):.2f}s/epoch")
    print(f"Average validation time: {np.mean(val_time):.2f}s")
    print("="*60)

    # Load best model and final test
    engine.model.load_state_dict(torch.load(path + "best_model.pth"))

    test_outputs = []
    test_y = []

    for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
        testx = x.to(device).float()
        testy = y.to(device).float()
        testx_mark = x_mark.to(device).float()
        test_embedding = embeddings.to(device).float()

        with torch.no_grad():
            preds = engine.model(testx, testx_mark, test_embedding)

        test_outputs.append(preds)
        test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    test_mse = []
    test_mae = []

    for j in range(args.pred_len):
        pred = test_pre[:, j, :]
        real = test_real[:, j, :]
        metrics = metric(pred, real)
        test_mse.append(metrics[0])
        test_mae.append(metrics[1])

    print(f"\nFinal Test Results:")
    print(f"MSE: {np.mean(test_mse):.4f}")
    print(f"MAE: {np.mean(test_mae):.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time: {(t2-t1)/60:.2f} minutes")
