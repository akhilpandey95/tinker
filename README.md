# Tinker RL: Disruption & Novelty Prediction

A toy RL environment for training language models to predict scientific impact metrics (disruption index, novelty/conventionality).

Designed for your blog post: [Tinker, smol-RL and QDoRA](https://akhilpandey95.github.io/notes/tinker/)

## Overview

This package implements an adversarial RL environment where:
- **Agent A (trainable)**: Predicts whether a paper is disruptive/consolidating and novel/conventional
- **Agent B (fixed)**: Challenges the prediction with counterarguments

This follows the [Twenty Questions pattern](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/multiplayer_rl/twenty_questions) from Tinker but adapted for science of science metrics.

## Files

```
tinker_disruption_rl/
├── disruption_novelty_dataset.py  # Dataset creation (OpenAlex API or synthetic)
├── tinker_disruption_env.py       # RL environments (Env, EnvGroupBuilder, RLDataset)
├── train_disruption.py            # Training script with Tinker
├── sample_dataset.jsonl           # 20-paper synthetic dataset for testing
└── README.md                      # This file
```

## Quick Start

### 1. Create Dataset

```bash
# Synthetic dataset (no API required) - good for testing
python disruption_novelty_dataset.py --synthetic --n_papers 200 --output data/toy_dataset.jsonl

# Real dataset from OpenAlex (requires httpx, tqdm)
pip install httpx tqdm
python disruption_novelty_dataset.py --n_papers 500 --email your-email@example.com --output data/real_dataset.jsonl
```

### 2. Test Environments

```python
import asyncio
from tinker_disruption_env import Paper, AdversarialDisruptionEnv

paper = Paper(
    openalex_id="test",
    title="Attention Is All You Need",
    abstract="We propose a new simple network architecture...",
    publication_year=2017,
    cited_by_count=50000,
    cd_index=0.65,
    novelty_score=0.82,
    conventionality_score=0.35,
    disruption_label="disruptive",
    novelty_label="novel",
    primary_field="Computer Science"
)

async def run():
    env = AdversarialDisruptionEnv(paper)
    obs, stop = await env.initial_observation()
    # ... agent generates response ...
    result = await env.step(action_tokens)
    print(f"Reward: {result.reward}")

asyncio.run(run())
```

### 3. Train with Tinker

```bash
python train_disruption.py --task adversarial --epochs 3 --batch_size 4
```

## Environment Types

| Environment | Description | Turns | Reward |
|-------------|-------------|-------|--------|
| `DisruptionPredictionEnv` | Single-turn disruption prediction | 1 | ±1.0 + 0.5 reasoning bonus |
| `AdversarialDisruptionEnv` | Multi-turn with challenger | 2-3 | ±1.0 + 0.3 reasoning + 0.2 adaptation |
| `NoveltyPredictionEnv` | Single-turn novelty/conventionality | 1 | ±1.0 (partial credit for adjacent) |
| `CombinedImpactEnv` | Both metrics simultaneously | 1 | 0.5 × disruption + 0.5 × novelty |

## Dataset Schema

```json
{
  "openalex_id": "W123456789",
  "title": "Paper Title",
  "abstract": "Full abstract text...",
  "publication_year": 2020,
  "cited_by_count": 150,
  "cd_index": 0.35,
  "novelty_score": 0.72,
  "conventionality_score": 0.45,
  "disruption_label": "disruptive",
  "novelty_label": "novel",
  "primary_field": "Computer Science",
  "concepts": ["machine learning", "deep learning"]
}
```

## Labels

**Disruption (CD Index)**:
- `disruptive` (CD > 0.1): Paper's citations don't cite its references
- `consolidating` (CD < -0.1): Paper's citations also cite its references  
- `neutral` (-0.1 ≤ CD ≤ 0.1): Developmental work

**Novelty** (Uzzi et al. 2013 style):
- `novel`: Atypical/unusual reference combinations
- `conventional`: Typical reference combinations
- `balanced`: Mix of both (often high-impact)

## Blog Post Integration

Add this to your "Just-RL, tinker and hello world !!" section:

```markdown
### Testing RLVR on Scientific Impact Prediction

The Just-RL paper [5] shows impressive results on reasoning tasks.
But can RLVR help with *scientific* evaluation tasks?

I built an adversarial environment based on Tinker's [Twenty Questions](...)
pattern. The agent must:
1. Predict if a paper is disruptive or consolidating
2. Defend its prediction against a challenger
3. Optionally revise based on valid counterarguments

This connects to my LMRSD work - can we train models to be better
calibrated on scientific impact?

[code snippet from train_disruption.py]
```

## Dependencies

```bash
# Core (for synthetic dataset and environments)
# No external dependencies required!

# For OpenAlex API dataset
pip install httpx tqdm

# For training with Tinker
pip install tinker  # or follow Tinker installation guide
```

## Extending

### Add Your Own Environment

```python
from tinker_disruption_env import Env, StepResult, StopCondition

class MyCustomEnv(Env):
    async def initial_observation(self):
        prompt = "Your prompt here..."
        return tokenize(prompt), StopCondition(max_tokens=200)
    
    async def step(self, action):
        response = detokenize(action)
        reward = compute_reward(response)
        return StepResult(reward=reward, done=True)
```

### Use SciSciNet-v2 Data

If you have access to your SciSciNet-v2 dataset locally:

```python
from disruption_novelty_dataset import load_from_sciscinet

records = load_from_sciscinet("path/to/sciscinet.parquet", n_papers=1000)
```

## References

- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Just-RL Paper](https://arxiv.org/pdf/2512.16649)
- [CD Index: Funk & Owen-Smith 2017](https://doi.org/10.1287/mnsc.2015.2366)
- [Novelty/Conventionality: Uzzi et al. 2013](https://doi.org/10.1126/science.1240474)
- [Your LMRSD Paper](https://akhilpandey95.github.io/projects/)

---

Author: Akhil Pandey Akella
License: MIT
