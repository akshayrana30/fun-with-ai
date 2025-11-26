import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        """Reshape tensor for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Final linear projection
        output = self.dense(context_layer)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = self.dense1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        feed_forward_output = self.feed_forward(hidden_states)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.layer_norm2(hidden_states + feed_forward_output)
        
        return output


class BERTEmbeddings(nn.Module):
    """BERT Embeddings: Token + Position + Segment embeddings"""
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, 
                 type_vocab_size=2, dropout_prob=0.1):
        super(BERTEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTEncoder(nn.Module):
    """Stack of Transformer Encoder Layers"""
    
    def __init__(self, num_layers, hidden_size, num_attention_heads, 
                 intermediate_size, dropout_prob=0.1):
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_attention_heads, 
                                   intermediate_size, dropout_prob)
            for _ in range(num_layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BERTPooler(nn.Module):
    """Pooler to get sentence representation from [CLS] token"""
    
    def __init__(self, hidden_size):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # Take [CLS] token (first token)
        first_token = hidden_states[:, 0]
        pooled_output = self.dense(first_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERT(nn.Module):
    """BERT Model for pre-training"""
    
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super(BERT, self).__init__()
        
        self.hidden_size = hidden_size
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, 
                                        max_position_embeddings, 
                                        type_vocab_size, dropout_prob)
        self.encoder = BERTEncoder(num_layers, hidden_size, num_attention_heads, 
                                   intermediate_size, dropout_prob)
        self.pooler = BERTPooler(hidden_size)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert attention mask to proper format
        # [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Pass through encoder
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        
        # Pool for sentence representation
        pooled_output = self.pooler(encoder_output)
        
        return encoder_output, pooled_output


class BERTForMaskedLM(nn.Module):
    """BERT model with Masked Language Modeling head"""
    
    def __init__(self, bert_model, vocab_size):
        super(BERTForMaskedLM, self).__init__()
        self.bert = bert_model
        self.mlm_head = nn.Linear(bert_model.hidden_size, vocab_size)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        encoder_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm_head(encoder_output)
        return mlm_logits


class BERTForNextSentencePrediction(nn.Module):
    """BERT model with Next Sentence Prediction head"""
    
    def __init__(self, bert_model):
        super(BERTForNextSentencePrediction, self).__init__()
        self.bert = bert_model
        self.nsp_head = nn.Linear(bert_model.hidden_size, 2)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        nsp_logits = self.nsp_head(pooled_output)
        return nsp_logits


class BERTForPreTraining(nn.Module):
    """BERT model with both MLM and NSP heads for pre-training"""
    
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
        super(BERTForPreTraining, self).__init__()
        
        self.bert = BERT(vocab_size, hidden_size, num_layers, num_attention_heads,
                        intermediate_size, max_position_embeddings, type_vocab_size, dropout_prob)
        
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        encoder_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        
        # MLM predictions
        mlm_logits = self.mlm_head(encoder_output)
        
        # NSP predictions
        nsp_logits = self.nsp_head(pooled_output)
        
        return mlm_logits, nsp_logits
