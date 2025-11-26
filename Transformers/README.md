# BERT Implementation in PyTorch

This folder contains a complete implementation of BERT (Bidirectional Encoder Representations from Transformers) from scratch using PyTorch.

## Overview

BERT is a transformer-based model designed for natural language understanding tasks. This implementation includes:

- **Multi-Head Self-Attention**: Core attention mechanism that allows the model to focus on different parts of the input
- **Transformer Encoder**: Stack of encoder layers with self-attention and feed-forward networks
- **BERT Embeddings**: Combination of token, position, and segment embeddings
- **Pre-training Heads**: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)

## Files

- `bert.py`: Complete BERT implementation with all components
- `demo.py`: Demo scripts showing how to use the BERT model
- `README.md`: This file

## Architecture Components

### 1. MultiHeadAttention
Implements the multi-head self-attention mechanism, allowing the model to attend to different representation subspaces.

### 2. FeedForward
Position-wise feed-forward network with GELU activation.

### 3. TransformerEncoderLayer
Single transformer encoder layer combining self-attention and feed-forward networks with residual connections and layer normalization.

### 4. BERTEmbeddings
Combines three types of embeddings:
- Token embeddings: Word piece embeddings
- Position embeddings: Positional information
- Segment embeddings: Sentence A/B distinction

### 5. BERTEncoder
Stack of transformer encoder layers.

### 6. BERT
Main BERT model combining embeddings, encoder, and pooler.

### 7. BERTForPreTraining
BERT with pre-training heads for MLM and NSP tasks.

## Usage

### Basic BERT Model

```python
import torch
from bert import BERT

# Create model
model = BERT(
    vocab_size=30522,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# Prepare input
input_ids = torch.randint(0, 30522, (2, 128))  # (batch_size, seq_length)
attention_mask = torch.ones((2, 128))
token_type_ids = torch.zeros((2, 128), dtype=torch.long)

# Forward pass
encoder_output, pooled_output = model(input_ids, token_type_ids, attention_mask)
```

### BERT for Pre-training

```python
from bert import BERTForPreTraining

# Create pre-training model
model = BERTForPreTraining(
    vocab_size=30522,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# Forward pass
mlm_logits, nsp_logits = model(input_ids, token_type_ids, attention_mask)
```

## Running the Demo

```bash
cd Transformers
python demo.py
```

The demo script will run several examples demonstrating:
1. Basic BERT model usage
2. BERT pre-training with MLM and NSP
3. Multi-head attention mechanism
4. Individual BERT components

## Model Configurations

### BERT-Base
- Hidden size: 768
- Number of layers: 12
- Attention heads: 12
- Intermediate size: 3072
- Total parameters: ~110M

### BERT-Large
- Hidden size: 1024
- Number of layers: 24
- Attention heads: 16
- Intermediate size: 4096
- Total parameters: ~340M

### Custom Configuration
You can create custom configurations by adjusting the parameters:

```python
model = BERT(
    vocab_size=30522,
    hidden_size=256,      # Smaller hidden size
    num_layers=6,         # Fewer layers
    num_attention_heads=8,
    intermediate_size=1024
)
```

## Key Features

- ✅ Complete BERT architecture implementation
- ✅ Multi-head self-attention mechanism
- ✅ Position and segment embeddings
- ✅ Layer normalization and residual connections
- ✅ GELU activation function
- ✅ Masked Language Modeling head
- ✅ Next Sentence Prediction head
- ✅ Attention masking support
- ✅ Configurable model size

## Requirements

```
torch>=1.7.0
```

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Notes

This is an educational implementation for learning purposes. For production use, consider using the official [Hugging Face Transformers](https://huggingface.co/transformers/) library which provides pre-trained weights and optimized implementations.
