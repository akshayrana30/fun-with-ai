"""
2. PPO (Proximal Policy Optimization) Training - RLHF Step 3

This script demonstrates how PPO is used to fine-tune language models with RL.
PPO is the algorithm that updates the model based on the reward model's feedback.

Key Concepts:
- PPO is a policy gradient method that's stable and sample-efficient
- It prevents large policy updates that could destabilize training
- Uses a "clipped" objective to limit how much the policy can change
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class PPOTrainer:
    """
    Simplified PPO trainer for language model fine-tuning.
    
    PPO Key Ideas:
    1. Collect experiences (prompts, responses, rewards)
    2. Compute advantages (how much better than expected)
    3. Update policy while limiting change (clipping)
    4. Add KL penalty to prevent drift from original model
    """
    
    def __init__(self, policy_model, reward_model, ref_model, tokenizer, device):
        """
        Args:
            policy_model: The model being trained (LLM)
            reward_model: Trained reward model for scoring outputs
            ref_model: Reference model (frozen copy) for KL penalty
            tokenizer: Tokenizer for the model
            device: CPU or CUDA
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Freeze reference model (it doesn't get updated)
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # PPO hyperparameters
        self.clip_epsilon = 0.2  # How much we allow policy to change
        self.kl_coef = 0.1       # Weight for KL divergence penalty
        self.gamma = 0.99        # Discount factor for rewards
        self.lam = 0.95          # GAE (Generalized Advantage Estimation) parameter
        
    def compute_advantages(self, rewards, values):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Advantages tell us how much better an action was than expected.
        Positive advantage = better than expected (increase probability)
        Negative advantage = worse than expected (decrease probability)
        
        Args:
            rewards: Rewards from the reward model
            values: Value estimates from a critic (simplified here)
            
        Returns:
            advantages: Advantages for each step
        """
        advantages = []
        advantage = 0
        
        # Compute advantages backward through time
        for t in reversed(range(len(rewards))):
            # Temporal difference error
            # In this simplified version, we use reward directly as value
            delta = rewards[t] - (values[t] if t < len(values) else 0)
            
            # Accumulate advantage with decay
            advantage = delta + self.gamma * self.lam * advantage
            advantages.insert(0, advantage)
            
        # Normalize advantages (makes training more stable)
        advantages = torch.tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def compute_kl_divergence(self, policy_logprobs, ref_logprobs):
        """
        Compute KL divergence between policy and reference model.
        
        KL divergence measures how different two probability distributions are.
        We penalize large KL to keep the fine-tuned model close to the original.
        
        Why? Prevents "reward hacking" where the model exploits the reward model
        in ways that break its general capabilities.
        
        Args:
            policy_logprobs: Log probabilities from current policy
            ref_logprobs: Log probabilities from reference model
            
        Returns:
            kl_div: KL divergence value
        """
        # KL(ref || policy) = sum(ref * log(ref/policy))
        # In log space: ref_logprob - policy_logprob
        kl_div = (ref_logprobs - policy_logprobs).mean()
        return kl_div
    
    def ppo_step(self, prompts, responses, old_logprobs, advantages, optimizer):
        """
        Perform one PPO update step.
        
        This is where the magic happens - we update the policy to:
        1. Maximize reward (via advantages)
        2. Stay close to the old policy (via clipping)
        3. Stay close to the reference model (via KL penalty)
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            old_logprobs: Log probs from the old policy (before update)
            advantages: Computed advantages
            optimizer: Optimizer for policy updates
        """
        # Get new log probabilities from current policy
        with torch.no_grad():
            # Tokenize prompt + response
            full_text = [p + r for p, r in zip(prompts, responses)]
            encoded = self.tokenizer(
                full_text,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
        
        # Forward pass through policy
        outputs = self.policy(**encoded)
        logits = outputs.logits
        
        # Compute log probabilities for generated tokens
        # (Simplified - in practice, this is more complex)
        new_logprobs = F.log_softmax(logits, dim=-1).mean(dim=1).mean(dim=-1)
        
        # Get reference model log probs for KL penalty
        with torch.no_grad():
            ref_outputs = self.ref_model(**encoded)
            ref_logits = ref_outputs.logits
            ref_logprobs = F.log_softmax(ref_logits, dim=-1).mean(dim=1).mean(dim=-1)
        
        # Compute probability ratio: π_new / π_old
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # PPO clipped objective
        # This is the key innovation of PPO!
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        # Take the minimum to prevent large updates
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL divergence penalty
        kl_div = self.compute_kl_divergence(new_logprobs, ref_logprobs)
        kl_penalty = self.kl_coef * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        # Update policy
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item()
        }


def demonstrate_ppo_concepts():
    """
    Demonstrate PPO concepts with simple examples.
    This is educational - not a full training loop.
    """
    print("=== PPO Training Demonstration ===\n")
    
    print("PPO Clipping Explanation:")
    print("-" * 50)
    print("PPO prevents large policy updates using clipping.")
    print("Example with clip_epsilon = 0.2:\n")
    
    # Simulate different scenarios
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    old_logprobs = torch.tensor([0.0, 0.0, 0.0, 0.0])
    new_logprobs_scenarios = [
        torch.tensor([0.1, 0.5, -0.1, -0.5]),  # Small, large positive changes
        torch.tensor([-0.1, -0.5, 0.1, 0.5])   # Small, large negative changes
    ]
    
    clip_epsilon = 0.2
    
    for i, new_logprobs in enumerate(new_logprobs_scenarios):
        print(f"\nScenario {i+1}:")
        ratio = torch.exp(new_logprobs - old_logprobs)
        print(f"Probability ratios: {ratio.numpy()}")
        
        # Unclipped objective
        surr1 = ratio * advantages
        print(f"Unclipped objective: {surr1.numpy()}")
        
        # Clipped objective
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        print(f"Clipped objective: {surr2.numpy()}")
        
        # PPO takes minimum (most conservative)
        ppo_objective = torch.min(surr1, surr2)
        print(f"PPO objective (min): {ppo_objective.numpy()}")
        print("-> Large updates are clipped, small updates pass through")
    
    print("\n" + "=" * 50)
    print("\nKL Divergence Penalty Explanation:")
    print("-" * 50)
    print("KL penalty keeps the model from drifting too far from the original.")
    print("This prevents 'reward hacking' and preserves general capabilities.\n")
    
    # Example KL values and their meaning
    kl_examples = [
        (0.01, "Very similar - safe update"),
        (0.1, "Moderate difference - acceptable"),
        (0.5, "Large difference - risky"),
        (1.0, "Very different - dangerous")
    ]
    
    kl_coef = 0.1
    for kl_val, description in kl_examples:
        penalty = kl_coef * kl_val
        print(f"KL = {kl_val:.2f}: {description}")
        print(f"  -> Penalty added to loss: {penalty:.3f}\n")
    
    print("=" * 50)
    print("\nAdvantage Explanation:")
    print("-" * 50)
    print("Advantages tell us how good an action was compared to average.")
    print("They guide which behaviors to reinforce.\n")
    
    example_scenarios = [
        ("High reward response", 2.5, 1.0, "+1.5", "Strongly reinforce"),
        ("Average response", 1.0, 1.0, "0.0", "No change"),
        ("Poor response", -0.5, 1.0, "-1.5", "Discourage")
    ]
    
    print(f"{'Scenario':<25} {'Reward':<10} {'Baseline':<10} {'Advantage':<12} {'Action'}")
    print("-" * 75)
    for scenario, reward, baseline, advantage, action in example_scenarios:
        print(f"{scenario:<25} {reward:<10.1f} {baseline:<10.1f} {advantage:<12} {action}")
    
    print("\n" + "=" * 50)


# Main execution
if __name__ == "__main__":
    print("=== RLHF PPO Training Demo ===\n")
    
    # Demonstrate concepts
    demonstrate_ppo_concepts()
    
    print("\n\n=== PPO in RLHF Pipeline ===")
    print("-" * 50)
    print("""
In actual RLHF training:

1. Generate responses using the policy model
2. Score responses with the reward model
3. Compute advantages (how good each response was)
4. Update policy with PPO:
   - Increase probability of high-advantage actions
   - Decrease probability of low-advantage actions
   - Clip large updates for stability
   - Add KL penalty to stay close to original model
5. Repeat for many iterations

The result: A model that generates responses humans prefer,
while maintaining its original capabilities and stability.
    """)
    
    print("\n=== Key Takeaways ===")
    print("1. PPO prevents large, unstable policy updates via clipping")
    print("2. KL penalty keeps the model from drifting too far")
    print("3. Advantages guide which behaviors to reinforce")
    print("4. This combination makes RLHF training stable and effective")
    
    print("\n[NOTE: This is an educational demonstration.]")
    print("[A full implementation requires substantial compute resources.]")
