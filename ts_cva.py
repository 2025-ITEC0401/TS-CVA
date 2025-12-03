"""
TS-CVA Training Wrapper

Provides a training interface similar to TS2Vec for the TS-CVA model.
Supports both pure time series training and cross-modal training with LLM embeddings.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from models import TSCVA, TSCVAEncoder
from models.losses import hierarchical_contrastive_loss
from models.ts_cva import cross_modal_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan


class TSCVAWrapper:
    """
    TS-CVA Training Wrapper
    
    Provides training interface compatible with TS2Vec pipeline while
    supporting cross-modal alignment with LLM embeddings.
    """
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        num_heads: int = 8,
        d_llm: int = 768,
        d_ff: int = 256,
        dropout: float = 0.1,
        device: str = 'cuda',
        lr: float = 0.001,
        batch_size: int = 16,
        max_train_length: int = None,
        temporal_unit: int = 0,
        use_cross_modal: bool = True,
        cross_modal_weight: float = 0.3,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        """
        Initialize TS-CVA model.
        
        Args:
            input_dims: Number of input features
            output_dims: Output representation dimension
            hidden_dims: Hidden dimension
            depth: Number of dilated conv blocks
            num_heads: Number of attention heads
            d_llm: LLM embedding dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            device: Device to use
            lr: Learning rate
            batch_size: Batch size
            max_train_length: Maximum sequence length for training
            temporal_unit: Minimum temporal unit for contrastive loss
            use_cross_modal: Whether to use cross-modal alignment
            cross_modal_weight: Weight for cross-modal loss (0-1)
            after_iter_callback: Callback after each iteration
            after_epoch_callback: Callback after each epoch
        """
        super().__init__()
        
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.use_cross_modal = use_cross_modal
        self.cross_modal_weight = cross_modal_weight
        self.output_dims = output_dims
        
        # Initialize model
        self._net = TSCVA(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            num_heads=num_heads,
            d_llm=d_llm,
            d_ff=d_ff,
            dropout=dropout,
            use_cross_modal=use_cross_modal
        ).to(self.device)
        
        # Exponential moving average model
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray = None,
        train_llm_embeddings: np.ndarray = None,
        test_llm_embeddings: np.ndarray = None,
        n_epochs: int = None,
        n_iters: int = None,
        verbose: bool = False
    ):
        """
        Train the TS-CVA model.
        
        Args:
            train_data: Training time series [n_samples, n_timestamps, n_features]
            test_data: Test time series for validation
            train_llm_embeddings: LLM embeddings for training [n_samples, n_tokens, d_llm]
            test_llm_embeddings: LLM embeddings for validation
            n_epochs: Number of epochs
            n_iters: Number of iterations
            verbose: Print training progress
            
        Returns:
            total_loss_log, loss_log, val_loss_log
        """
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600
        
        # Handle long sequences
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
                if train_llm_embeddings is not None:
                    # Repeat LLM embeddings for each section
                    train_llm_embeddings = np.tile(train_llm_embeddings, (sections, 1, 1))
        
        # Handle variable length series
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
        
        # Remove fully NaN samples
        valid_mask = ~np.isnan(train_data).all(axis=2).all(axis=1)
        train_data = train_data[valid_mask]
        if train_llm_embeddings is not None:
            train_llm_embeddings = train_llm_embeddings[valid_mask]
        
        if test_data is not None:
            test_valid_mask = ~np.isnan(test_data).all(axis=2).all(axis=1)
            test_data = test_data[test_valid_mask]
            if test_llm_embeddings is not None:
                test_llm_embeddings = test_llm_embeddings[test_valid_mask]
        
        # Create data loaders
        if train_llm_embeddings is not None:
            train_dataset = TensorDataset(
                torch.from_numpy(train_data).float(),
                torch.from_numpy(train_llm_embeddings).float()
            )
        else:
            train_dataset = TensorDataset(torch.from_numpy(train_data).float())
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True
        )
        
        if test_data is not None:
            if test_llm_embeddings is not None:
                test_dataset = TensorDataset(
                    torch.from_numpy(test_data).float(),
                    torch.from_numpy(test_llm_embeddings).float()
                )
            else:
                test_dataset = TensorDataset(torch.from_numpy(test_data).float())
            test_loader = DataLoader(
                test_dataset,
                batch_size=min(self.batch_size, len(test_dataset)),
                shuffle=False,
                drop_last=True
            )
        else:
            test_loader = None
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        val_loss_log = []
        total_loss_log = []
        epoch_time = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            epoch_start_time = time.time()
            cum_loss = 0
            cum_val_loss = 0
            n_epoch_iters = 0
            n_val_iters = 0
            
            interrupted = False
            self._net.train()
            
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]
                llm_emb = batch[1] if len(batch) > 1 else None
                
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset:window_offset + self.max_train_length]
                
                x = x.to(self.device)
                if llm_emb is not None:
                    llm_emb = llm_emb.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                # Two augmented views
                x1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                x2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                
                out1 = self._net(x1, llm_emb)
                out1 = out1[:, -crop_l:]
                
                out2 = self._net(x2, llm_emb)
                out2 = out2[:, :crop_l]
                
                # Hierarchical contrastive loss
                loss_ts = hierarchical_contrastive_loss(
                    out1, out2,
                    temporal_unit=self.temporal_unit
                )
                
                # Cross-modal contrastive loss
                if self.use_cross_modal and llm_emb is not None:
                    # Handle 4D LLM embeddings [B, E, D, C] -> [B, E, D]
                    if llm_emb.dim() == 4:
                        llm_emb_3d = llm_emb.mean(dim=-1)
                    else:
                        llm_emb_3d = llm_emb
                    
                    # Pool representations for cross-modal loss
                    ts_pooled = F.adaptive_avg_pool1d(
                        out1.transpose(1, 2), 1
                    ).squeeze(-1)  # [B, D]
                    llm_pooled = F.adaptive_avg_pool1d(
                        llm_emb_3d.transpose(1, 2), 1
                    ).squeeze(-1)  # [B, d_llm]
                    
                    # Project LLM to same dim if needed
                    if llm_pooled.size(-1) != ts_pooled.size(-1):
                        llm_pooled = self._net.encoder.llm_projection(llm_pooled)
                    
                    loss_cross = cross_modal_contrastive_loss(ts_pooled, llm_pooled)
                    loss = (1 - self.cross_modal_weight) * loss_ts + self.cross_modal_weight * loss_cross
                else:
                    loss = loss_ts
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            # Validation
            if test_loader is not None:
                self._net.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        x = batch[0].to(self.device)
                        llm_emb = batch[1].to(self.device) if len(batch) > 1 else None
                        
                        # Handle 4D LLM embeddings for validation too
                        if llm_emb is not None and llm_emb.dim() == 4:
                            llm_emb_val = llm_emb.mean(dim=-1)
                        else:
                            llm_emb_val = llm_emb
                        
                        out = self._net(x, llm_emb)
                        val_loss = hierarchical_contrastive_loss(
                            out, out,
                            temporal_unit=self.temporal_unit
                        )
                        cum_val_loss += val_loss.item()
                        n_val_iters += 1
            
            if interrupted:
                break
            
            # Logging
            if n_epoch_iters > 0:
                cum_loss /= n_epoch_iters
                loss_log.append(cum_loss)
            
            if n_val_iters > 0:
                cum_val_loss /= n_val_iters
                val_loss_log.append(cum_val_loss)
            else:
                val_loss_log.append(cum_loss)
            
            total_loss = (cum_loss + (cum_val_loss if n_val_iters > 0 else cum_loss)) / 2
            total_loss_log.append(total_loss)
            
            # Save best model
            current_val = val_loss_log[-1]
            if not hasattr(self, '_best_val_loss') or current_val < self._best_val_loss:
                self._best_val_loss = current_val
                self._best_epoch = self.n_epochs
                self._best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                if verbose:
                    print(f"[NEW BEST] ", end='')
            
            if verbose:
                print(f"Epoch #{self.n_epochs}: train_loss={cum_loss:.6f}, "
                      f"val_loss={val_loss_log[-1]:.6f}", end=', ')
            
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss, val_loss_log[-1])
            
            epoch_end_time = time.time()
            epoch_time.append(epoch_end_time - epoch_start_time)
            if len(epoch_time) > 10:
                epoch_time.pop(0)
            
            if verbose:
                remaining = np.mean(epoch_time) * (n_epochs - self.n_epochs) / 60 if n_epochs else float('inf')
                print(f"time={epoch_end_time - epoch_start_time:.2f}s, remaining={remaining:.2f}min")
        
        return total_loss_log, loss_log, val_loss_log
    
    def encode(
        self,
        data: np.ndarray,
        llm_embeddings: np.ndarray = None,
        encoding_window: str = None,
        batch_size: int = None
    ) -> np.ndarray:
        """
        Encode time series data.
        
        Args:
            data: Time series [n_samples, n_timestamps, n_features]
            llm_embeddings: Optional LLM embeddings
            encoding_window: 'full_series' for pooled representation
            batch_size: Batch size for encoding
            
        Returns:
            representations: Encoded representations
        """
        assert self.net is not None, 'Please train or load a model first'
        assert data.ndim == 3
        
        if batch_size is None:
            batch_size = self.batch_size
        
        org_training = self.net.training
        self.net.eval()
        
        if llm_embeddings is not None:
            dataset = TensorDataset(
                torch.from_numpy(data).float(),
                torch.from_numpy(llm_embeddings).float()
            )
        else:
            dataset = TensorDataset(torch.from_numpy(data).float())
        
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            outputs = []
            for batch in loader:
                x = batch[0].to(self.device)
                llm_emb = batch[1].to(self.device) if len(batch) > 1 else None
                
                out = self.net(x, llm_emb)
                
                if encoding_window == 'full_series':
                    out = F.max_pool1d(
                        out.transpose(1, 2),
                        kernel_size=out.size(1)
                    ).transpose(1, 2).squeeze(1)
                
                outputs.append(out.cpu())
            
            output = torch.cat(outputs, dim=0)
        
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn: str):
        """Save model to file."""
        torch.save(self.net.state_dict(), fn)
    
    def save_best(self, fn: str):
        """Save best model to file."""
        if hasattr(self, '_best_state'):
            torch.save(self._best_state, fn)
            print(f"Saved best model from epoch {self._best_epoch} (val_loss={self._best_val_loss:.6f})")
        else:
            print("No best model found, saving current model")
            self.save(fn)
    
    def load_best(self):
        """Load best model state (call after fit)."""
        if hasattr(self, '_best_state'):
            self._net.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})
            self.net.update_parameters(self._net)
            print(f"Loaded best model from epoch {self._best_epoch} (val_loss={self._best_val_loss:.6f})")
        else:
            print("No best model found")
    
    def load(self, fn: str):
        """Load model from file."""
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
