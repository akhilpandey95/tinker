# Sub-Agent Runbook

This folder contains:
- `prompts/`: one prompt per worktree branch.
- `orchestration_commands.md`: launch + git orchestration commands.

## Branches and Worktrees
- `main` -> `/Users/akhilpandey/code/writing/tinker`
- `feat/data-pipeline` -> `/Users/akhilpandey/code/writing/tinker_wt/data`
- `feat/env-validation` -> `/Users/akhilpandey/code/writing/tinker_wt/env`
- `feat/baseline-sft` -> `/Users/akhilpandey/code/writing/tinker_wt/sft`
- `feat/rlvr-training` -> `/Users/akhilpandey/code/writing/tinker_wt/rlvr`
- `feat/repro-stability` -> `/Users/akhilpandey/code/writing/tinker_wt/stability`
- `feat/comparison` -> `/Users/akhilpandey/code/writing/tinker_wt/report`

## Suggested Execution Order
1. `data`, `env`, `stability`
2. merge `data` -> `main`
3. rebase `env`, `sft`, `rlvr`, `stability` on updated `main`
4. merge `env` -> `main`
5. run and merge `sft` and `rlvr`
6. merge `stability`
7. rebase/run/merge `report`

## Alignment Documents
Every sub-agent prompt references:
- `rh/content/notes/tinker.md`
- `rh/content/notes/tinker_part2.md`
- `tinker/README.md`
- `tinker/exp_init.md`
