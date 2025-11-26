import torch
from bert import BERT, BERTForPreTraining, MultiHeadAttention, BERTEmbeddings, TransformerEncoderLayer


def demo_basic_bert():
    """Demo of basic BERT model usage"""
    print("=" * 60)
    print("Demo: Basic BERT Model")
    print("=" * 60)
    
    # Model configuration
    vocab_size = 30522  # Similar to BERT-base
    hidden_size = 768
    num_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    max_position_embeddings = 512
    
    # Create BERT model
    model = BERT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings
    )
    
    print(f"Created BERT model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        encoder_output, pooled_output = model(input_ids, token_type_ids, attention_mask)
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Pooled output shape: {pooled_output.shape}")
    print()


def demo_bert_pretraining():
    """Demo of BERT pre-training model with MLM and NSP"""
    print("=" * 60)
    print("Demo: BERT Pre-training (MLM + NSP)")
    print("=" * 60)
    
    # Smaller configuration for demo
    vocab_size = 30522
    hidden_size = 256
    num_layers = 4
    num_attention_heads = 4
    intermediate_size = 1024
    
    # Create BERT pre-training model
    model = BERTForPreTraining(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size
    )
    
    print(f"Created BERT pre-training model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    token_type_ids = torch.cat([
        torch.zeros((batch_size, seq_length // 2), dtype=torch.long),
        torch.ones((batch_size, seq_length // 2), dtype=torch.long)
    ], dim=1)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        mlm_logits, nsp_logits = model(input_ids, token_type_ids, attention_mask)
    
    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"NSP logits shape: {nsp_logits.shape}")
    print()


def demo_attention_visualization():
    """Demo showing attention mechanism works"""
    print("=" * 60)
    print("Demo: Attention Mechanism")
    print("=" * 60)
    
    hidden_size = 768
    num_attention_heads = 12
    batch_size = 1
    seq_length = 10
    
    attention = MultiHeadAttention(hidden_size, num_attention_heads)
    
    # Create dummy input
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Forward pass
    attention.eval()
    with torch.no_grad():
        output = attention(hidden_states)
    
    print(f"Attention output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in attention.parameters())}")
    print()


def demo_model_components():
    """Demo individual components of BERT"""
    print("=" * 60)
    print("Demo: BERT Components")
    print("=" * 60)
    
    vocab_size = 30522
    hidden_size = 768
    batch_size = 2
    seq_length = 32
    
    # Test embeddings
    print("Testing BERTEmbeddings...")
    embeddings = BERTEmbeddings(vocab_size, hidden_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    
    embeddings.eval()
    with torch.no_grad():
        emb_output = embeddings(input_ids, token_type_ids)
    print(f"  Embedding output shape: {emb_output.shape}")
    
    # Test encoder layer
    print("Testing TransformerEncoderLayer...")
    encoder_layer = TransformerEncoderLayer(
        hidden_size=hidden_size,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    encoder_layer.eval()
    with torch.no_grad():
        layer_output = encoder_layer(emb_output)
    print(f"  Encoder layer output shape: {layer_output.shape}")
    print()


if __name__ == "__main__":
    print("\n")
    print("#" * 60)
    print("# BERT Implementation Demo")
    print("# PyTorch implementation of BERT from scratch")
    print("#" * 60)
    print("\n")
    
    # Run demos
    demo_basic_bert()
    demo_bert_pretraining()
    demo_attention_visualization()
    demo_model_components()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
