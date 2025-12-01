"""
1. Reward Model Training - RLHF Step 2

This script demonstrates how to build and train a reward model for RLHF.
A reward model learns to predict human preferences between different text outputs.

Key Concepts:
- Reward models take text as input and output a scalar score
- They're trained on pairwise comparisons (which output is better?)
- The model learns to approximate human judgment
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np


class RewardModel(nn.Module):
    """
    A simple reward model that scores text based on quality/preference.
    
    Architecture:
    - Uses a pre-trained transformer (e.g., BERT) as encoder
    - Adds a regression head to output a single scalar reward
    """
    
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        # Load pre-trained transformer
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder layers (optional - faster training)
        # In practice, you might want to fine-tune some layers
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Add reward head: transformer hidden size -> single scalar
        hidden_size = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Output single reward score
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass: encode text and predict reward
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for padding
            
        Returns:
            reward: Scalar reward score for the input text
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Predict reward score
        reward = self.reward_head(cls_output)
        
        return reward.squeeze(-1)  # Remove last dimension


class PreferenceDataset(Dataset):
    """
    Dataset for pairwise preference data.
    
    Format: Each example has:
    - prompt: The input/question
    - chosen: The preferred response
    - rejected: The non-preferred response
    """
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine prompt with each response
        chosen_text = item['prompt'] + " " + item['chosen']
        rejected_text = item['prompt'] + " " + item['rejected']
        
        # Tokenize both texts
        chosen_encoded = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoded['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoded['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoded['attention_mask'].squeeze(0)
        }


def train_reward_model(model, dataloader, optimizer, device, epochs=3):
    """
    Train the reward model using pairwise ranking loss.
    
    Loss Function:
    - We want reward(chosen) > reward(rejected)
    - Use ranking loss: -log(sigmoid(reward_chosen - reward_rejected))
    - This encourages the model to give higher scores to preferred outputs
    """
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move data to device
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)
            
            # Get reward predictions
            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)
            
            # Ranking loss: we want chosen_reward > rejected_reward
            # Using log-sigmoid for numerical stability
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # The loss should decrease over time as the model learns preferences
        
    return model


def evaluate_reward_model(model, test_texts, tokenizer, device):
    """
    Evaluate the reward model on sample texts.
    Shows how the model scores different types of responses.
    """
    model.eval()
    
    print("\n=== Reward Model Evaluation ===")
    
    with torch.no_grad():
        for text in test_texts:
            # Tokenize
            encoded = tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Get reward
            reward = model(encoded['input_ids'], encoded['attention_mask'])
            
            print(f"\nText: {text}")
            print(f"Reward Score: {reward.item():.4f}")


# Example usage
if __name__ == "__main__":
    print("=== RLHF Reward Model Training Demo ===\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create sample preference data
    # In practice, this would come from human labelers ranking model outputs
    sample_data = [
        {
            'prompt': 'Explain quantum computing in simple terms.',
            'chosen': 'Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, allowing for much faster computation on certain problems.',
            'rejected': 'Quantum computing is complicated and uses physics stuff.'
        },
        {
            'prompt': 'What is the capital of France?',
            'chosen': 'The capital of France is Paris, a historic city known for the Eiffel Tower and the Louvre Museum.',
            'rejected': 'France has a capital.'
        },
        {
            'prompt': 'How do I bake a cake?',
            'chosen': 'To bake a cake: 1) Mix dry ingredients (flour, sugar, baking powder), 2) Add wet ingredients (eggs, milk, oil), 3) Pour into a pan, 4) Bake at 350°F for 30-35 minutes.',
            'rejected': 'Just put stuff in the oven.'
        }
    ]
    
    # Initialize tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    reward_model = RewardModel("bert-base-uncased").to(device)
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(sample_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    
    # Train the reward model
    print("\nTraining reward model...")
    trained_model = train_reward_model(reward_model, dataloader, optimizer, device, epochs=5)
    
    # Evaluate on test examples
    test_texts = [
        "Paris is the capital of France and is famous for its art and culture.",
        "I don't know.",
        "Quantum computers use quantum mechanics to solve complex problems efficiently."
    ]
    
    evaluate_reward_model(trained_model, test_texts, tokenizer, device)
    
    print("\n=== Key Takeaways ===")
    print("1. Reward models learn to score text based on human preferences")
    print("2. Training uses pairwise comparisons (chosen vs rejected)")
    print("3. The model learns to give higher scores to better responses")
    print("4. This reward model can then guide RL fine-tuning of LLMs")
