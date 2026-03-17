from __future__ import annotations

# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import argparse
import asyncio

import matplotlib.pyplot as plt

from util import build_disruption_rl_config, plot_rl_metrics, run_disruption_rl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--renderer-name", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    return parser.parse_args()


async def main():
    args = parse_args()
    cfg_kwargs = {
        key: value
        for key, value in {
            "model_name": args.model_name,
            "renderer_name": args.renderer_name,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "log_path": args.log_path,
        }.items()
        if value is not None
    }
    cfg = build_disruption_rl_config(**cfg_kwargs)
    log_path = await run_disruption_rl(cfg)
    metrics_path = log_path / "metrics.jsonl"
    fig = plot_rl_metrics(metrics_path, show=False)
    fig.savefig(log_path / "rl_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    asyncio.run(main())
