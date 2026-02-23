# Tinker RL: Disruption & Novelty Prediction

A toy RL environment for training language models to predict scientific impact metrics (disruption index, novelty/conventionality).

Blog post: [Tinker, smol-RL and QDoRA](https://akhilpandey95.github.io/notes/tinker/)

## Overview

This package implements an adversarial RL environment where:
- **Agent A (trainable)**: Predicts whether a paper is disruptive/consolidating and novel/conventional
- **Agent B (fixed)**: Challenges the prediction with counterarguments

This follows the [Twenty Questions pattern](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/multiplayer_rl/twenty_questions) from Tinker but adapted for science of science metrics.

## Files

```
tinker_disruption_rl/
├── disruption_novelty_dataset.py  # Dataset creation (synthetic, OpenAlex, SciSciNet parquet)
├── tinker_disruption_env.py       # RL environments (Env, EnvGroupBuilder, RLDataset)
├── train_disruption.py            # Training script with Tinker
├── sample_dataset.jsonl           # 20-paper synthetic dataset for testing
└── README.md                      # This file
```

## Quick Start

### 1. Create Dataset

```bash
# Synthetic dataset (no API required) - good for testing
python disruption_novelty_dataset.py --synthetic --n-papers 200 --output data/toy_dataset.jsonl

# SciSciNet parquet dataset (alpha parquet with title + abstract_inverted_index)
python disruption_novelty_dataset.py \
  --n-papers 500000 \
  --tinker-sciscinet-parquet tinker_sciscinet_papers_alpha.parquet \
  --parquet-primary tinker \
  --sciscinet-language en \
  --output data/sciscinet/disruption_novelty_sciscinet.jsonl \
  --splits-output data/sciscinet/disruption_novelty_sciscinet.splits.json \
  --metadata-output data/sciscinet/disruption_novelty_sciscinet.metadata.json

# Real dataset from OpenAlex (requires httpx, tqdm)
pip install httpx tqdm
python disruption_novelty_dataset.py --n-papers 500 --email your-email@example.com --output data/real_dataset.jsonl
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

## SciSciNet Alpha Input

The parquet loader now supports your merged alpha file:

- `tinker_sciscinet_papers_alpha.parquet` (~38,851,803 rows)
- core columns:
  `paperid`, `doi`, `year`, `date`, `doctype`, `reference_count`, `citation_count`,
  `C3`, `C5`, `C10`, `disruption`, `Atyp_Median_Z`, `Atyp_10pct_Z`, `Atyp_Pairs`,
  `title`, `abstract_inverted_index`, `language`

How these map into training JSONL:

- `openalex_id` <- `paperid`
- `cd_index` <- `disruption`
- `cited_by_count` <- `cited_by_count` if present, else `citation_count`
- `title` <- `title`
- `abstract` <- decoded from `abstract_inverted_index` (JSON/dict string) when plain abstract text is absent
- novelty/conventionality are derived from Uzzi-style atypicality columns when explicit normalized scores are absent
- `primary_field` / `concepts` may be placeholders when no rich field assignment table is joined (for example, `"Article"`)

## Paper-Field Alpha Enrichment (Next Step)

A second parquet is now available for enriching `primary_field` and `concepts` with
hierarchical field assignments after the base dataset is generated:

- `tinker_sciscinet_paper_fields_alpha.parquet` (~6.3G)
- observed row count: `274,920,196`
- schema:
  - `paperid`
  - `fieldid`
  - `doi`
  - `display_name`
  - `level`
  - `description`
  - `title`

This parquet is a paper-to-field link table (many-to-many). It should be used to
upgrade the current placeholder field metadata in the base JSONL (especially for
`primary_field` and `concepts`) without rerunning the full 500k dataset build.

Recommended workflow:

1. Build the base disruption/novelty JSONL from `tinker_sciscinet_papers_alpha.parquet` (current pipeline).
2. Run a post-enrichment step that:
   - lazily scans `tinker_sciscinet_paper_fields_alpha.parquet`
   - semi-joins on the base dataset paper IDs
   - aggregates field rows per `paperid`
   - writes an enriched JSONL (new output path, do not overwrite by default)
3. Recompute label distributions by `primary_field` on the enriched dataset.

Suggested deterministic mapping:

- `primary_field`: prefer `level == 0`, else lowest `level`, tie-break alphabetically
- `concepts`: unique `display_name` values sorted deterministically, truncated to top `k` (for example `8`)

Field hierarchy references (OpenAlex concepts):
- https://docs.openalex.org/api-entities/concepts
- https://docs.openalex.org/api-entities/concepts/concept-object

## Practical 500k Run (Local SciSciNet Alpha)

Below is a practical launch command and observed runtime/memory profile for the
current local-only dataset builder on an EC2 `ml.m7i.8xlarge` (128 GB RAM, 32 vCPU).

### Launch Command (500k scan target)

```bash
cd /home/ec2-user/SageMaker/sciscinetv2/tinker

POLARS_MAX_THREADS=8 /usr/bin/time -v python tinker_disruption_rl/disruption_novelty_dataset.py \
  --n-papers 500000 \
  --tinker-sciscinet-parquet ../tinker_sciscinet_papers_alpha.parquet \
  --sciscinet-language en \
  --batch-size 1000 \
  --row-log-interval 50000 \
  --output data/sciscinet/disruption_novelty_sciscinet_500k.jsonl \
  --splits-output data/sciscinet/disruption_novelty_sciscinet_500k.splits.json \
  --metadata-output data/sciscinet/disruption_novelty_sciscinet_500k.metadata.json
```

### Observed Output (Practical Run)

```text
mode=sciscinet_parquet records=491820 output=data/sciscinet/disruption_novelty_sciscinet_500k.jsonl
splits train=393456 val=49182 test=49182
split_ids_sha256=359d7c9d722da4e2ceeefaaffb7673655ba7373971ff82adcd3ce7fb80300123
```

### `/usr/bin/time -v` (Key Lines)

```text
User time (seconds): 142.35
System time (seconds): 19.79
Percent of CPU this job got: 94%
Elapsed (wall clock) time (h:mm:ss or m:ss): 2:50.79
Maximum resident set size (kbytes): 67253592
Major (requiring I/O) page faults: 53073
Minor (reclaiming a frame) page faults: 12824865
File system inputs: 6955128
File system outputs: 1689856
Exit status: 0
```

### Artifact Sizes (Practical Run)

```text
data/sciscinet/disruption_novelty_sciscinet_500k.jsonl      816M
data/sciscinet/disruption_novelty_sciscinet_500k.splits.json 9.9M
data/sciscinet/disruption_novelty_sciscinet_500k.metadata.json 1.6K
```

Notes:
- `--n-papers 500000` is a scan/selection limit before row-level normalization; the final JSONL may contain fewer valid rows.
- In the run above, `500000` selected rows produced `491820` normalized records.
- `POLARS_MAX_THREADS=8` and `--batch-size 1000` materially reduced memory pressure versus more aggressive settings.

## Labels

**Disruption (CD Index)**:
- `disruptive` (CD > 0.1): Paper's citations don't cite its references
- `consolidating` (CD < -0.1): Paper's citations also cite its references  
- `neutral` (-0.1 ≤ CD ≤ 0.1): Developmental work

**Novelty** (Uzzi et al. 2013 style):
- `novel`: Atypical/unusual reference combinations
- `conventional`: Typical reference combinations
- `balanced`: Mix of both (often high-impact)

SciSciNet parquet mode uses:
- `disruption` column as `cd_index` (Wu et al. 2019)
- `Atyp_10pct_Z` as atypicality/novelty signal
- `Atyp_Median_Z` as conventionality signal
- `Atyp_Pairs` as supporting context

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

# For SciSciNet parquet ingestion
pip install polars pyarrow

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

records = load_from_sciscinet(
    "sciscinet_papers.parquet",
    tinker_sciscinet_parquet="tinker_sciscinet_papers_alpha.parquet",
    n_papers=1000,
    parquet_primary="tinker",
    sciscinet_language="en",
)
```

## References

- [Thinking Machines Lab: Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Just-RL (arXiv:2512.16649)](https://arxiv.org/abs/2512.16649)
- [Funk & Owen-Smith (2017), CD Index](https://doi.org/10.1287/mnsc.2015.2366)
- [Uzzi et al. (2013), Atypical Combinations and Scientific Impact](https://doi.org/10.1126/science.1240474)
- [AP Akella, HV Siravuri, S Rohatgi (2025), "Pre-review to Peer review: Pitfalls of Automating Reviews using Large Language Models"](https://arxiv.org/abs/2512.22145)

---

Author: Akhil Pandey Akella
License: MIT
