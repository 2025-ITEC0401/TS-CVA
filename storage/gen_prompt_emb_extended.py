"""
Extended Prompt Embedding Generator with External Information Support

This module extends GenPromptEmbUEA to support external contextual information
such as news, events, or domain-specific metadata in the prompt generation.

Example use cases:
- Stock market: Include news headlines, earnings reports, market sentiment
- Weather: Include seasonal information, geographic context
- Healthcare: Include patient demographics, medical history summaries
- IoT sensors: Include device metadata, location, environment conditions
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from typing import Optional, Dict, List, Union
import json


class GenPromptEmbExtended(nn.Module):
    """
    Generate GPT-2 embeddings from time series data with external information.
    
    The prompt template can include:
    1. Time series statistics (min, max, mean, std, trend)
    2. External context (news, events, metadata)
    3. Domain-specific information
    
    Args:
        device: CUDA device
        input_len: Length of input time series
        n_features: Number of features/channels
        dataset_name: Name of the dataset
        model_name: Hugging Face model name (default: gpt2)
        d_model: Model hidden dimension (768 for GPT-2)
        fixed_token_len: Fixed token length for consistent output shape
        feature_names: Optional list of feature/channel names
        domain: Domain type for specialized prompts ('finance', 'weather', 'health', 'general')
    """
    
    def __init__(
        self,
        device='cuda',
        input_len=100,
        n_features=6,
        dataset_name='Dataset',
        model_name='gpt2',
        d_model=768,
        fixed_token_len=128,  # Larger for external info
        feature_names: Optional[List[str]] = None,
        domain: str = 'general'
    ):
        super(GenPromptEmbExtended, self).__init__()
        
        self.device = device
        self.input_len = input_len
        self.n_features = n_features
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.d_model = d_model
        self.fixed_token_len = fixed_token_len
        self.domain = domain
        
        # Feature names for interpretability
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"channel_{i+1}" for i in range(n_features)]
        
        # Load GPT-2
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze GPT-2 parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"GPT-2 loaded. Domain: {domain}, Fixed token len: {fixed_token_len}")
    
    def _compute_statistics(self, x: torch.Tensor) -> Dict:
        """Compute statistics for a time series."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        stats = {
            'min': x.min(dim=0)[0],
            'max': x.max(dim=0)[0],
            'mean': x.mean(dim=0),
            'std': x.std(dim=0),
        }
        
        # Trend analysis
        quarter = max(1, x.shape[0] // 4)
        first_quarter_mean = x[:quarter].mean(dim=0)
        last_quarter_mean = x[-quarter:].mean(dim=0)
        stats['trend'] = last_quarter_mean - first_quarter_mean
        
        total_trend = stats['trend'].sum()
        if total_trend > 0.1:
            stats['trend_dir'] = 'increasing'
        elif total_trend < -0.1:
            stats['trend_dir'] = 'decreasing'
        else:
            stats['trend_dir'] = 'stable'
        
        # Volatility (for finance)
        if x.shape[0] > 1:
            returns = (x[1:] - x[:-1]) / (x[:-1].abs() + 1e-8)
            stats['volatility'] = returns.std(dim=0)
        else:
            stats['volatility'] = torch.zeros(x.shape[1], device=x.device)
        
        return stats
    
    def _create_finance_prompt(
        self,
        x: torch.Tensor,
        channel_idx: Optional[int] = None,
        external_info: Optional[Dict] = None
    ) -> str:
        """Create prompt for financial time series with news/events."""
        stats = self._compute_statistics(x)
        
        # Base financial statistics
        if channel_idx is not None:
            feature_name = self.feature_names[channel_idx]
            base_prompt = (
                f"Financial time series: {feature_name}. "
                f"Period: {self.input_len} timesteps. "
                f"Price range: {stats['min'][channel_idx]:.2f} to {stats['max'][channel_idx]:.2f}. "
                f"Average: {stats['mean'][channel_idx]:.2f}, Volatility: {stats['volatility'][channel_idx]:.4f}. "
                f"Trend: {stats['trend_dir']}. "
            )
        else:
            mean_vol = stats['volatility'].mean().item()
            base_prompt = (
                f"Multi-asset financial data from {self.dataset_name}. "
                f"{self.n_features} assets, {self.input_len} periods. "
                f"Average volatility: {mean_vol:.4f}. "
                f"Overall trend: {stats['trend_dir']}. "
            )
        
        # Add external information
        if external_info:
            context_parts = []
            
            # News headlines
            if 'news' in external_info:
                news = external_info['news']
                if isinstance(news, list):
                    news = '; '.join(news[:3])  # Top 3 headlines
                context_parts.append(f"Recent news: {news}")
            
            # Market sentiment
            if 'sentiment' in external_info:
                sentiment = external_info['sentiment']
                context_parts.append(f"Market sentiment: {sentiment}")
            
            # Key events
            if 'events' in external_info:
                events = external_info['events']
                if isinstance(events, list):
                    events = ', '.join(events[:2])
                context_parts.append(f"Key events: {events}")
            
            # Economic indicators
            if 'indicators' in external_info:
                indicators = external_info['indicators']
                if isinstance(indicators, dict):
                    ind_str = ', '.join([f"{k}: {v}" for k, v in list(indicators.items())[:3]])
                    context_parts.append(f"Indicators: {ind_str}")
            
            if context_parts:
                base_prompt += ' '.join(context_parts)
        
        return base_prompt
    
    def _create_weather_prompt(
        self,
        x: torch.Tensor,
        channel_idx: Optional[int] = None,
        external_info: Optional[Dict] = None
    ) -> str:
        """Create prompt for weather/climate time series."""
        stats = self._compute_statistics(x)
        
        if channel_idx is not None:
            feature_name = self.feature_names[channel_idx]
            base_prompt = (
                f"Weather measurement: {feature_name}. "
                f"Duration: {self.input_len} observations. "
                f"Range: {stats['min'][channel_idx]:.2f} to {stats['max'][channel_idx]:.2f}. "
                f"Mean: {stats['mean'][channel_idx]:.2f}. "
                f"Trend: {stats['trend_dir']}. "
            )
        else:
            base_prompt = (
                f"Multivariate weather data from {self.dataset_name}. "
                f"{self.n_features} measurements, {self.input_len} observations. "
                f"Trend: {stats['trend_dir']}. "
            )
        
        # Add external context
        if external_info:
            context_parts = []
            
            if 'season' in external_info:
                context_parts.append(f"Season: {external_info['season']}")
            
            if 'location' in external_info:
                context_parts.append(f"Location: {external_info['location']}")
            
            if 'forecast' in external_info:
                context_parts.append(f"Forecast: {external_info['forecast']}")
            
            if 'alerts' in external_info:
                context_parts.append(f"Weather alerts: {external_info['alerts']}")
            
            if context_parts:
                base_prompt += ' '.join(context_parts)
        
        return base_prompt
    
    def _create_health_prompt(
        self,
        x: torch.Tensor,
        channel_idx: Optional[int] = None,
        external_info: Optional[Dict] = None
    ) -> str:
        """Create prompt for healthcare/biomedical time series."""
        stats = self._compute_statistics(x)
        
        if channel_idx is not None:
            feature_name = self.feature_names[channel_idx]
            base_prompt = (
                f"Biomedical signal: {feature_name}. "
                f"Recording duration: {self.input_len} samples. "
                f"Signal range: {stats['min'][channel_idx]:.2f} to {stats['max'][channel_idx]:.2f}. "
                f"Baseline: {stats['mean'][channel_idx]:.2f}, Variability: {stats['std'][channel_idx]:.2f}. "
                f"Pattern: {stats['trend_dir']}. "
            )
        else:
            base_prompt = (
                f"Multi-channel biomedical recording from {self.dataset_name}. "
                f"{self.n_features} signals, {self.input_len} samples. "
                f"Overall pattern: {stats['trend_dir']}. "
            )
        
        # Add patient/clinical context
        if external_info:
            context_parts = []
            
            if 'patient_info' in external_info:
                info = external_info['patient_info']
                if isinstance(info, dict):
                    info_str = ', '.join([f"{k}: {v}" for k, v in info.items()])
                    context_parts.append(f"Patient: {info_str}")
                else:
                    context_parts.append(f"Patient: {info}")
            
            if 'diagnosis' in external_info:
                context_parts.append(f"Diagnosis: {external_info['diagnosis']}")
            
            if 'medications' in external_info:
                meds = external_info['medications']
                if isinstance(meds, list):
                    meds = ', '.join(meds)
                context_parts.append(f"Medications: {meds}")
            
            if 'clinical_notes' in external_info:
                context_parts.append(f"Notes: {external_info['clinical_notes']}")
            
            if context_parts:
                base_prompt += ' '.join(context_parts)
        
        return base_prompt
    
    def _create_general_prompt(
        self,
        x: torch.Tensor,
        channel_idx: Optional[int] = None,
        external_info: Optional[Dict] = None
    ) -> str:
        """Create general-purpose prompt."""
        stats = self._compute_statistics(x)
        
        if channel_idx is not None:
            feature_name = self.feature_names[channel_idx]
            base_prompt = (
                f"Time series from {self.dataset_name}, {feature_name}. "
                f"Length: {self.input_len}. "
                f"Range: {stats['min'][channel_idx]:.2f} to {stats['max'][channel_idx]:.2f}. "
                f"Mean: {stats['mean'][channel_idx]:.2f}, Std: {stats['std'][channel_idx]:.2f}. "
                f"Trend: {stats['trend_dir']}. "
            )
        else:
            base_prompt = (
                f"Multivariate time series from {self.dataset_name}. "
                f"{self.n_features} channels, {self.input_len} timesteps. "
                f"Overall trend: {stats['trend_dir']}. "
            )
        
        # Add any external context
        if external_info:
            context_parts = []
            
            if 'description' in external_info:
                context_parts.append(external_info['description'])
            
            if 'context' in external_info:
                context_parts.append(f"Context: {external_info['context']}")
            
            if 'metadata' in external_info:
                meta = external_info['metadata']
                if isinstance(meta, dict):
                    meta_str = ', '.join([f"{k}: {v}" for k, v in meta.items()])
                    context_parts.append(f"Metadata: {meta_str}")
            
            if context_parts:
                base_prompt += ' '.join(context_parts)
        
        return base_prompt
    
    def create_prompt(
        self,
        x: torch.Tensor,
        channel_idx: Optional[int] = None,
        external_info: Optional[Dict] = None
    ) -> str:
        """
        Create a text prompt based on domain and external information.
        
        Args:
            x: Time series tensor [T, C]
            channel_idx: If specified, create prompt for single channel
            external_info: Dictionary with external contextual information
                
        Returns:
            Text prompt string
        """
        if self.domain == 'finance':
            return self._create_finance_prompt(x, channel_idx, external_info)
        elif self.domain == 'weather':
            return self._create_weather_prompt(x, channel_idx, external_info)
        elif self.domain == 'health':
            return self._create_health_prompt(x, channel_idx, external_info)
        else:
            return self._create_general_prompt(x, channel_idx, external_info)
    
    def _get_embedding(self, prompt: str) -> torch.Tensor:
        """Get GPT-2 embedding for a text prompt."""
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tokens)
            embeddings = outputs.last_hidden_state
        
        return embeddings
    
    def generate_embeddings(
        self,
        x: torch.Tensor,
        external_info: Optional[Union[Dict, List[Dict]]] = None,
        mode: str = 'per_channel'
    ) -> torch.Tensor:
        """
        Generate embeddings for a batch of time series with external info.
        
        Args:
            x: Input tensor [B, T, C]
            external_info: External context - either single dict (same for all samples)
                          or list of dicts (one per sample)
            mode: 'per_channel' or 'summary'
                
        Returns:
            Embeddings tensor with fixed size
        """
        batch_size = x.shape[0]
        n_channels = x.shape[2]
        fixed_len = self.fixed_token_len
        
        # Normalize external_info to per-sample list
        if external_info is None:
            ext_info_list = [None] * batch_size
        elif isinstance(external_info, dict):
            ext_info_list = [external_info] * batch_size
        else:
            ext_info_list = external_info
            assert len(ext_info_list) == batch_size
        
        if mode == 'per_channel':
            result = torch.zeros(batch_size, fixed_len, self.d_model, n_channels, device=self.device)
            
            for b in range(batch_size):
                for c in range(n_channels):
                    prompt = self.create_prompt(x[b], channel_idx=c, external_info=ext_info_list[b])
                    emb = self._get_embedding(prompt)
                    token_len = min(emb.shape[1], fixed_len)
                    
                    result[b, :token_len, :, c] = emb[0, :token_len, :]
                    if token_len < fixed_len:
                        result[b, token_len:, :, c] = emb[0, -1:, :].expand(fixed_len - token_len, -1)
            
            return result
        
        else:  # summary mode
            result = torch.zeros(batch_size, fixed_len, self.d_model, device=self.device)
            
            for b in range(batch_size):
                prompt = self.create_prompt(x[b], external_info=ext_info_list[b])
                emb = self._get_embedding(prompt)
                token_len = min(emb.shape[1], fixed_len)
                
                result[b, :token_len, :] = emb[0, :token_len, :]
                if token_len < fixed_len:
                    result[b, token_len:, :] = emb[0, -1:, :].expand(fixed_len - token_len, -1)
            
            return result
    
    def forward(
        self,
        x: torch.Tensor,
        external_info: Optional[Union[Dict, List[Dict]]] = None,
        mode: str = 'summary'
    ) -> torch.Tensor:
        """Forward pass - alias for generate_embeddings."""
        return self.generate_embeddings(x, external_info, mode)


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("Extended Prompt Embedding Generator - Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy time series
    batch_size = 2
    seq_len = 100
    n_features = 3
    x = torch.randn(batch_size, seq_len, n_features).to(device)
    
    # Example 1: Finance domain with news
    print("\n1. Finance Domain with News:")
    print("-" * 40)
    
    gen_finance = GenPromptEmbExtended(
        device=device,
        input_len=seq_len,
        n_features=n_features,
        dataset_name='StockData',
        domain='finance',
        feature_names=['AAPL', 'GOOGL', 'MSFT'],
        fixed_token_len=128
    )
    
    finance_external = {
        'news': [
            'Fed announces interest rate decision',
            'Tech stocks rally on earnings beat',
            'Market volatility increases amid uncertainty'
        ],
        'sentiment': 'bullish',
        'events': ['Earnings season', 'FOMC meeting'],
        'indicators': {
            'VIX': 18.5,
            'DXY': 104.2,
            '10Y_yield': 4.25
        }
    }
    
    # Show example prompt
    prompt = gen_finance.create_prompt(x[0], channel_idx=0, external_info=finance_external)
    print(f"Example prompt:\n{prompt}\n")
    
    # Generate embeddings
    emb = gen_finance.generate_embeddings(x, external_info=finance_external, mode='summary')
    print(f"Embedding shape: {emb.shape}")
    
    # Example 2: Weather domain
    print("\n2. Weather Domain:")
    print("-" * 40)
    
    gen_weather = GenPromptEmbExtended(
        device=device,
        input_len=seq_len,
        n_features=n_features,
        dataset_name='WeatherStation',
        domain='weather',
        feature_names=['Temperature', 'Humidity', 'Pressure'],
        fixed_token_len=128
    )
    
    weather_external = {
        'season': 'Summer',
        'location': 'Seoul, South Korea',
        'forecast': 'Heavy rain expected in 24 hours',
        'alerts': 'Heat wave warning'
    }
    
    prompt = gen_weather.create_prompt(x[0], channel_idx=0, external_info=weather_external)
    print(f"Example prompt:\n{prompt}\n")
    
    # Example 3: Healthcare domain
    print("\n3. Healthcare Domain:")
    print("-" * 40)
    
    gen_health = GenPromptEmbExtended(
        device=device,
        input_len=seq_len,
        n_features=n_features,
        dataset_name='ECG',
        domain='health',
        feature_names=['Lead_I', 'Lead_II', 'Lead_III'],
        fixed_token_len=128
    )
    
    health_external = {
        'patient_info': {'age': 65, 'gender': 'M', 'weight': 75},
        'diagnosis': 'Suspected arrhythmia',
        'medications': ['Aspirin', 'Metoprolol'],
        'clinical_notes': 'Patient reported palpitations'
    }
    
    prompt = gen_health.create_prompt(x[0], channel_idx=0, external_info=health_external)
    print(f"Example prompt:\n{prompt}\n")
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
