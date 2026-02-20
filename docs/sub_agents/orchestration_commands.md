# Orchestration Commands (Worktrees + Sub-Agents)

## 1) Preflight Checks
Run from `/Users/akhilpandey/code/writing`:

```bash
# Verify worktrees
ls -la tinker tinker_wt
git -C tinker worktree list

# Verify clean state before starting
git -C tinker status --short --branch
git -C tinker_wt/data status --short --branch
git -C tinker_wt/env status --short --branch
git -C tinker_wt/sft status --short --branch
git -C tinker_wt/rlvr status --short --branch
git -C tinker_wt/stability status --short --branch
git -C tinker_wt/report status --short --branch
```

## 2) Prompt Locations
- `tinker/docs/sub_agents/prompts/01_data.md`
- `tinker/docs/sub_agents/prompts/02_env.md`
- `tinker/docs/sub_agents/prompts/03_sft.md`
- `tinker/docs/sub_agents/prompts/04_rlvr.md`
- `tinker/docs/sub_agents/prompts/05_stability.md`
- `tinker/docs/sub_agents/prompts/06_report.md`

## 3) Launch Order
Launch first wave in parallel:
1. `data`
2. `env`
3. `stability`

Launch second wave after merges/rebases:
1. `sft`
2. `rlvr`

Launch final wave last:
1. `report`

## 3A) Launch Sub-Agents with Codex CLI
Run from `/Users/akhilpandey/code/writing`:

```bash
# One-time setup
mkdir -p tinker/.agent_runs/events tinker/.agent_runs/final

# Optional: ensure Codex auth is valid
codex login
```

### Wave 1 (run in parallel)
```bash
codex exec --full-auto -C /Users/akhilpandey/code/writing/tinker_wt/data --json -o /Users/akhilpandey/code/writing/tinker/.agent_runs/final/data.txt - < /Users/akhilpandey/code/writing/tinker/docs/sub_agents/prompts/01_data.md > /Users/akhilpandey/code/writing/tinker/.agent_runs/events/data.jsonl 2>&1 &
echo $! > /Users/akhilpandey/code/writing/tinker/.agent_runs/data.pid

codex exec --full-auto -C /Users/akhilpandey/code/writing/tinker_wt/env --json -o /Users/akhilpandey/code/writing/tinker/.agent_runs/final/env.txt - < /Users/akhilpandey/code/writing/tinker/docs/sub_agents/prompts/02_env.md > /Users/akhilpandey/code/writing/tinker/.agent_runs/events/env.jsonl 2>&1 &
echo $! > /Users/akhilpandey/code/writing/tinker/.agent_runs/env.pid

codex exec --full-auto -C /Users/akhilpandey/code/writing/tinker_wt/stability --json -o /Users/akhilpandey/code/writing/tinker/.agent_runs/final/stability.txt - < /Users/akhilpandey/code/writing/tinker/docs/sub_agents/prompts/05_stability.md > /Users/akhilpandey/code/writing/tinker/.agent_runs/events/stability.jsonl 2>&1 &
echo $! > /Users/akhilpandey/code/writing/tinker/.agent_runs/stability.pid
```

### Wave 2 (after rebases/merges from wave 1)
```bash
codex exec --full-auto -C /Users/akhilpandey/code/writing/tinker_wt/sft --json -o /Users/akhilpandey/code/writing/tinker/.agent_runs/final/sft.txt - < /Users/akhilpandey/code/writing/tinker/docs/sub_agents/prompts/03_sft.md > /Users/akhilpandey/code/writing/tinker/.agent_runs/events/sft.jsonl 2>&1 &
echo $! > /Users/akhilpandey/code/writing/tinker/.agent_runs/sft.pid

codex exec --full-auto -C /Users/akhilpandey/code/writing/tinker_wt/rlvr --json -o /Users/akhilpandey/code/writing/tinker/.agent_runs/final/rlvr.txt - < /Users/akhilpandey/code/writing/tinker/docs/sub_agents/prompts/04_rlvr.md > /Users/akhilpandey/code/writing/tinker/.agent_runs/events/rlvr.jsonl 2>&1 &
echo $! > /Users/akhilpandey/code/writing/tinker/.agent_runs/rlvr.pid
```

### Wave 3 (last)
```bash
codex exec --full-auto -C /Users/akhilpandey/code/writing/tinker_wt/report --json -o /Users/akhilpandey/code/writing/tinker/.agent_runs/final/report.txt - < /Users/akhilpandey/code/writing/tinker/docs/sub_agents/prompts/06_report.md > /Users/akhilpandey/code/writing/tinker/.agent_runs/events/report.jsonl 2>&1 &
echo $! > /Users/akhilpandey/code/writing/tinker/.agent_runs/report.pid
```

### Monitor and collect outputs
```bash
# Check running processes
ps -p $(cat /Users/akhilpandey/code/writing/tinker/.agent_runs/*.pid | tr '\n' ' ') -o pid=,etime=,command=

# Tail event streams
tail -n 40 -f /Users/akhilpandey/code/writing/tinker/.agent_runs/events/data.jsonl
tail -n 40 -f /Users/akhilpandey/code/writing/tinker/.agent_runs/events/env.jsonl

# Read final message from each agent
sed -n '1,220p' /Users/akhilpandey/code/writing/tinker/.agent_runs/final/data.txt
sed -n '1,220p' /Users/akhilpandey/code/writing/tinker/.agent_runs/final/env.txt
sed -n '1,220p' /Users/akhilpandey/code/writing/tinker/.agent_runs/final/stability.txt
```

## 4) Rebase Commands (After `main` Moves)
Rebase each lane branch on latest `main` before continuing work or before merge.

```bash
# Update main first
cd /Users/akhilpandey/code/writing/tinker
git checkout main
git pull --ff-only || true

# Rebase lanes onto main
git -C /Users/akhilpandey/code/writing/tinker_wt/data checkout feat/data-pipeline
git -C /Users/akhilpandey/code/writing/tinker_wt/data rebase main

git -C /Users/akhilpandey/code/writing/tinker_wt/env checkout feat/env-validation
git -C /Users/akhilpandey/code/writing/tinker_wt/env rebase main

git -C /Users/akhilpandey/code/writing/tinker_wt/sft checkout feat/baseline-sft
git -C /Users/akhilpandey/code/writing/tinker_wt/sft rebase main

git -C /Users/akhilpandey/code/writing/tinker_wt/rlvr checkout feat/rlvr-training
git -C /Users/akhilpandey/code/writing/tinker_wt/rlvr rebase main

git -C /Users/akhilpandey/code/writing/tinker_wt/stability checkout feat/repro-stability
git -C /Users/akhilpandey/code/writing/tinker_wt/stability rebase main

git -C /Users/akhilpandey/code/writing/tinker_wt/report checkout feat/comparison
git -C /Users/akhilpandey/code/writing/tinker_wt/report rebase main
```

If conflict occurs in a worktree:

```bash
# Resolve files manually, then:
git add <resolved-files>
git rebase --continue

# Or abort if needed:
git rebase --abort
```

## 5) Merge Sequence (Recommended)
Run merges from `/Users/akhilpandey/code/writing/tinker` on `main`.

```bash
cd /Users/akhilpandey/code/writing/tinker
git checkout main

# 1) data
git merge --no-ff feat/data-pipeline -m "merge: data pipeline lane"

# 2) env (after rebase)
git merge --no-ff feat/env-validation -m "merge: env validation lane"

# 3) sft and rlvr (order can be swapped if independent)
git merge --no-ff feat/baseline-sft -m "merge: sft baseline lane"
git merge --no-ff feat/rlvr-training -m "merge: rlvr training lane"

# 4) stability
git merge --no-ff feat/repro-stability -m "merge: repro stability lane"

# 5) report last
git merge --no-ff feat/comparison -m "merge: final comparison lane"
```

## 6) Suggested Orchestration Cycle
Use this loop for each lane completion:

```bash
# A) Confirm lane branch is clean after commit
git -C <lane_worktree> status --short --branch

# B) Rebase lane onto current main
git -C <lane_worktree> rebase main

# C) Merge lane into main
git -C /Users/akhilpandey/code/writing/tinker checkout main
git -C /Users/akhilpandey/code/writing/tinker merge --no-ff <lane_branch> -m "merge: <lane_name>"

# D) Rebase remaining active lanes onto new main
git -C <other_lane_worktree> rebase main
```

## 7) Final Verification
```bash
# Ensure no pending changes anywhere
git -C tinker status --short --branch
git -C tinker_wt/data status --short --branch
git -C tinker_wt/env status --short --branch
git -C tinker_wt/sft status --short --branch
git -C tinker_wt/rlvr status --short --branch
git -C tinker_wt/stability status --short --branch
git -C tinker_wt/report status --short --branch

# Optional: compact graph
git -C tinker log --oneline --decorate --graph --max-count=30
```

## 8) Optional Cleanup (After Everything Is Merged)
```bash
# Remove external worktrees once done
git -C tinker worktree remove /Users/akhilpandey/code/writing/tinker_wt/data
git -C tinker worktree remove /Users/akhilpandey/code/writing/tinker_wt/env
git -C tinker worktree remove /Users/akhilpandey/code/writing/tinker_wt/sft
git -C tinker worktree remove /Users/akhilpandey/code/writing/tinker_wt/rlvr
git -C tinker worktree remove /Users/akhilpandey/code/writing/tinker_wt/stability
git -C tinker worktree remove /Users/akhilpandey/code/writing/tinker_wt/report

# Delete feature branches once merged
git -C tinker branch -d feat/data-pipeline
git -C tinker branch -d feat/env-validation
git -C tinker branch -d feat/baseline-sft
git -C tinker branch -d feat/rlvr-training
git -C tinker branch -d feat/repro-stability
git -C tinker branch -d feat/comparison
```
