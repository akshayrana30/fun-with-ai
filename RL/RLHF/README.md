# RLHF (Reinforcement Learning from Human Feedback)

## What is RLHF?

RLHF is a technique used to fine-tune Large Language Models (LLMs) to align them better with human preferences. It's the key technology behind ChatGPT and other advanced conversational AI systems.

## The RLHF Pipeline

The RLHF process typically involves three main stages:

### 1. Supervised Fine-Tuning (SFT)
- Start with a pre-trained language model
- Fine-tune it on high-quality demonstrations written by humans
- This creates an initial policy model that can follow instructions

### 2. Reward Model Training
- Collect comparison data: humans rank multiple model outputs for the same prompt
- Train a reward model to predict which outputs humans prefer
- The reward model learns to score outputs based on human preferences

### 3. Reinforcement Learning Fine-Tuning
- Use the reward model to fine-tune the policy using RL algorithms (typically PPO)
- The model learns to generate outputs that maximize the reward
- This aligns the model's behavior with human preferences

## Scripts in this Folder

1. **`1_reward_model.py`** - Demonstrates how to create and train a simple reward model
2. **`2_ppo_trainer.py`** - Shows how PPO (Proximal Policy Optimization) works for RL fine-tuning
3. **`3_rlhf_pipeline.py`** - Puts it all together in a complete RLHF pipeline example
4. **`requirements.txt`** - Required Python packages

## Key Concepts

### Reward Model
- Takes text input and outputs a scalar reward score
- Trained on human preference data
- Acts as a proxy for human judgment

### PPO (Proximal Policy Optimization)
- A policy gradient RL algorithm
- Ensures training stability by limiting policy updates
- Prevents the model from deviating too much from the original behavior

### KL Divergence Penalty
- Keeps the fine-tuned model close to the original model
- Prevents "reward hacking" where the model exploits the reward model
- Maintains the model's general capabilities

## Learning Resources

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - Original RLHF paper from OpenAI
- [Hugging Face TRL Library](https://github.com/huggingface/trl) - Tools for training LLMs with RL
- [Anthropic's RLHF Overview](https://www.anthropic.com/index/core-views-on-ai-safety)

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the scripts in order:
   ```bash
   python 1_reward_model.py
   python 2_ppo_trainer.py
   python 3_rlhf_pipeline.py
   ```

Each script includes detailed comments explaining the concepts and implementation.
