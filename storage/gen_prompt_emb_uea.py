"""
Generate LLM Prompt Embeddings for UEA/UCR Time Series Classification datasets

This module creates text prompts from time series data and generates
GPT-2 embeddings for cross-modal alignment in TS-CVA.
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np


class GenPromptEmbUEA(nn.Module):
    """
    Generate GPT-2 embeddings from time series data for UEA/UCR datasets.
    
    The prompt template describes the time series statistically:
    - Min, max, mean values
    - Trend direction
    - Variability
    
    Args:
        device: CUDA device
        input_len: Length of input time series
        n_features: Number of features/channels
        dataset_name: Name of the dataset
        model_name: Hugging Face model name (default: gpt2)
        d_model: Model hidden dimension (768 for GPT-2)
        fixed_token_len: Fixed token length for consistent output shape (default: 64)
    """
    
    def __init__(
        self,
        device='cuda',
        input_len=100,
        n_features=6,
        dataset_name='BasicMotions',
        model_name='gpt2',
        d_model=768,
        fixed_token_len=64
    ):
        super(GenPromptEmbUEA, self).__init__()
        
        self.device = device
        self.input_len = input_len
        self.n_features = n_features
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.d_model = d_model
        self.fixed_token_len = fixed_token_len  # Fixed length for consistent batching
        
        # Load GPT-2
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze GPT-2 parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"GPT-2 loaded. Hidden dim: {d_model}, Fixed token len: {fixed_token_len}")
    
    def _compute_statistics(self, x):
        """
        Compute statistics for a single time series.
        
        Args:
            x: Time series tensor [T, C] or [T]
            
        Returns:
            Dictionary of statistics
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        stats = {}
        
        # Per-channel statistics
        stats['min'] = x.min(dim=0)[0]
        stats['max'] = x.max(dim=0)[0]
        stats['mean'] = x.mean(dim=0)
        stats['std'] = x.std(dim=0)
        
        # Trend (difference between last and first quarter means)
        quarter = max(1, x.shape[0] // 4)
        first_quarter_mean = x[:quarter].mean(dim=0)
        last_quarter_mean = x[-quarter:].mean(dim=0)
        stats['trend'] = last_quarter_mean - first_quarter_mean
        
        # Overall trend direction
        total_trend = stats['trend'].sum()
        if total_trend > 0.1:
            stats['trend_dir'] = 'increasing'
        elif total_trend < -0.1:
            stats['trend_dir'] = 'decreasing'
        else:
            stats['trend_dir'] = 'stable'
        
        return stats
    
    def _create_prompt(self, x, channel_idx=None):
        """
        Create a text prompt from time series data.
        
        Args:
            x: Time series tensor [T, C]
            channel_idx: If specified, create prompt for single channel
            
        Returns:
            Text prompt string
        """
        stats = self._compute_statistics(x)
        
        if channel_idx is not None:
            # Single channel prompt
            prompt = (
                f"Time series from {self.dataset_name} dataset, channel {channel_idx+1}. "
                f"Length: {self.input_len} timesteps. "
                f"Values range from {stats['min'][channel_idx]:.2f} to {stats['max'][channel_idx]:.2f}, "
                f"with mean {stats['mean'][channel_idx]:.2f} and std {stats['std'][channel_idx]:.2f}. "
                f"The trend is {stats['trend_dir']} with change of {stats['trend'][channel_idx]:.2f}."
            )
        else:
            # Multi-channel summary prompt
            n_channels = x.shape[1]
            mean_of_means = stats['mean'].mean().item()
            mean_of_stds = stats['std'].mean().item()
            
            prompt = (
                f"Multivariate time series from {self.dataset_name} dataset. "
                f"{n_channels} channels, {self.input_len} timesteps. "
                f"Overall mean: {mean_of_means:.2f}, variability: {mean_of_stds:.2f}. "
                f"Trend: {stats['trend_dir']}."
            )
        
        return prompt
    
    def _get_embedding(self, prompt):
        """
        Get GPT-2 embedding for a text prompt.
        
        Args:
            prompt: Text string
            
        Returns:
            Embedding tensor [1, seq_len, d_model]
        """
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tokens)
            embeddings = outputs.last_hidden_state
        
        return embeddings
    
    def generate_embeddings(self, x, mode='per_channel'):
        """
        Generate embeddings for a batch of time series.
        
        Args:
            x: Input tensor [B, T, C]
            mode: 'per_channel' or 'summary'
                - per_channel: One embedding per channel
                - summary: One embedding for entire series
                
        Returns:
            Embeddings tensor (always fixed size for consistent batching)
            - per_channel: [B, fixed_token_len, d_model, C]
            - summary: [B, fixed_token_len, d_model]
        """
        batch_size = x.shape[0]
        n_channels = x.shape[2]
        fixed_len = self.fixed_token_len
        
        if mode == 'per_channel':
            # Generate embedding for each channel with fixed output size
            result = torch.zeros(batch_size, fixed_len, self.d_model, n_channels, device=self.device)
            
            for b in range(batch_size):
                for c in range(n_channels):
                    prompt = self._create_prompt(x[b], channel_idx=c)
                    emb = self._get_embedding(prompt)  # [1, tokens, d_model]
                    token_len = min(emb.shape[1], fixed_len)
                    
                    # Copy tokens (truncate if longer than fixed_len)
                    result[b, :token_len, :, c] = emb[0, :token_len, :]
                    
                    # Pad with last token embedding if shorter
                    if token_len < fixed_len:
                        result[b, token_len:, :, c] = emb[0, -1:, :].expand(fixed_len - token_len, -1)
            
            return result
        
        else:  # summary mode
            result = torch.zeros(batch_size, fixed_len, self.d_model, device=self.device)
            
            for b in range(batch_size):
                prompt = self._create_prompt(x[b])
                emb = self._get_embedding(prompt)  # [1, tokens, d_model]
                token_len = min(emb.shape[1], fixed_len)
                
                # Copy tokens (truncate if longer than fixed_len)
                result[b, :token_len, :] = emb[0, :token_len, :]
                
                # Pad with last token embedding if shorter
                if token_len < fixed_len:
                    result[b, token_len:, :] = emb[0, -1:, :].expand(fixed_len - token_len, -1)
            
            return result
    
    def forward(self, x, mode='summary'):
        """Forward pass - alias for generate_embeddings."""
        return self.generate_embeddings(x, mode=mode)
