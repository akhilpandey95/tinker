from __future__ import annotations

import json
import os
import sys
import random
import argparse
import subprocess
import importlib.util
from pathlib import Path

SEED = 2026
random.seed(SEED)
WORKDIR = Path.cwd()
#LLM = '/home/ubuntu/models/Qwen3.5-9B'
LLM = '/home/ubuntu/models/Qwen3-8B'
DEBUG_MAX_STEPS = 250
DEBUG_VAL_SIZE = 200
DEBUG_TEST_SIZE = 100
RAW_COMPLETION_PREVIEW_COUNT = 5
os.environ["HF_TOKEN"] = "oops"

print('python', sys.version)
print('workdir', WORKDIR)
subprocess.run(['nvidia-smi'], check=False)

DATASET_PATH = Path('../data/sci_balanced_from2m_no_ovr.rl_balanced.jsonl')
SPLITS_PATH = Path('../data/sci_balanced_from2m_no_ovr.splits.json')

required_paths = [DATASET_PATH, SPLITS_PATH]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise FileNotFoundError(f'Missing required local dataset files: {missing}')

for path in required_paths:
    print(path.name, 'exists, size_mb=', round(path.stat().st_size / 1024**2, 2))

TRAINER_PATH = WORKDIR / 'qdora_train_local.py'
print('trainer_size_kb =', round(TRAINER_PATH.stat().st_size / 1024, 1))

spec = importlib.util.spec_from_file_location('qdora_train_local', TRAINER_PATH)
qdora_train_local = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qdora_train_local
spec.loader.exec_module(qdora_train_local)

print('default_dataset_path =', qdora_train_local.default_dataset_path())
print('default_splits_path  =', qdora_train_local.default_splits_path())
print('default_output_dir   =', qdora_train_local.default_output_dir())


def print_prediction_preview(path: Path, split_name: str, limit: int) -> None:
    if limit <= 0:
        return
    if not path.exists():
        print(f'{split_name}_preview_file_missing =', path)
        return

    print(f'{split_name}_preview_file =', path)
    shown = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            completion = str(row.get('completion_raw', '')).replace('\n', '\\n')
            if len(completion) > 220:
                completion = completion[:217] + '...'
            print(
                f'{split_name}_preview_{shown + 1:02d} =',
                {
                    'gold_label': row.get('gold_label'),
                    'pred_label': row.get('pred_label'),
                    'raw_status': row.get('raw_status'),
                    'clean_status': row.get('clean_status'),
                    'label_match': row.get('label_match'),
                    'completion_raw': completion,
                },
            )
            shown += 1
            if shown >= limit:
                break
    print(f'{split_name}_preview_count =', shown)

CONFIG = {
    'model_name': LLM,
    'dataset_path': DATASET_PATH,
    'splits_path': SPLITS_PATH,
    'split_source': 'seeded_jsonl',
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test',
    'train_size': 100_000,
    'val_size': DEBUG_VAL_SIZE,
    'test_size': DEBUG_TEST_SIZE,
    'output_dir': WORKDIR / 'qdora_sft_runs' / 'colab_h100_qwen3_8b_debug_250step',
    'overwrite_output_dir': True,
    'seed': 0,
    'max_length': 1024,
    'eval_max_new_tokens': 64,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 16,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1.0,
    'max_steps': DEBUG_MAX_STEPS,
    'learning_rate': 1e-5,
    'weight_decay': 0.1,
    'warmup_ratio': None,
    'warmup_steps': 15,
    'lr_scheduler_type': 'cosine',
    'max_grad_norm': 1.0,
    'logging_steps': 10,
    'save_strategy': 'no',
    'save_steps': 250,
    'save_total_limit': 2,
    'dataloader_num_workers': 4,
    'dataloader_pin_memory': True,
    'dataloader_persistent_workers': True,
    'group_by_length': True,
    'pad_to_multiple_of': 8,
    'gradient_checkpointing': False,
    'trust_remote_code': False,
    'attn_implementation': 'flash_attention_2',
    'lora_rank': 64,
    'lora_alpha': 128,
    'lora_dropout': 0.05,
    'target_modules': ','.join(qdora_train_local.DEFAULT_TARGET_MODULES),
    'teacher_confidence_mode': 'cd_index_margin',
    'report_to': 'none',
    'resume_from_checkpoint': None,
}

args = argparse.Namespace(**CONFIG)
print(CONFIG)

qdora_train_local.set_seed(args.seed)
qdora_train_local.ensure_output_dir(args.output_dir, overwrite=args.overwrite_output_dir)

tokenizer = qdora_train_local.load_tokenizer(args)
split_to_records = qdora_train_local.load_experiment_record_splits(args)
train_records = split_to_records['train']
val_records = split_to_records['val']
test_records = split_to_records['test']

raw_train_examples = qdora_train_local.build_sft_examples(train_records, tokenizer, args.teacher_confidence_mode)
raw_val_examples = qdora_train_local.build_sft_examples(val_records, tokenizer, args.teacher_confidence_mode)
raw_test_examples = qdora_train_local.build_sft_examples(test_records, tokenizer, args.teacher_confidence_mode)

train_examples, train_filter_stats = qdora_train_local.filter_examples_by_length(tokenizer, raw_train_examples, args.max_length)
val_examples, val_filter_stats = qdora_train_local.filter_examples_by_length(tokenizer, raw_val_examples, args.max_length)
test_examples, test_filter_stats = qdora_train_local.filter_examples_by_length(tokenizer, raw_test_examples, args.max_length)

print('split_source       =', args.split_source)
print('train_records      =', len(train_records))
print('val_records        =', len(val_records))
print('test_records       =', len(test_records))
print('train_examples     =', len(train_examples))
print('val_examples       =', len(val_examples))
print('test_examples      =', len(test_examples))
print('train_filter_stats =', train_filter_stats)
print('val_filter_stats   =', val_filter_stats)
print('test_filter_stats  =', test_filter_stats)

model, model_summary = qdora_train_local.load_model(args)
trainer, train_metrics = qdora_train_local.train_model(
    model=model,
    tokenizer=tokenizer,
    train_examples=train_examples,
    args=args,
    output_dir=args.output_dir,
)

qdora_train_local.prepare_model_for_eval(trainer.model)
val_metrics = qdora_train_local.evaluate_split(
    model=trainer.model,
    tokenizer=tokenizer,
    examples=val_examples,
    split_name='val',
    args=args,
    output_dir=args.output_dir,
)

test_metrics = qdora_train_local.evaluate_split(
    model=trainer.model,
    tokenizer=tokenizer,
    examples=test_examples,
    split_name='test',
    args=args,
    output_dir=args.output_dir,
)

print_prediction_preview(
    args.output_dir / 'predictions_val.jsonl',
    split_name='val',
    limit=RAW_COMPLETION_PREVIEW_COUNT,
)
print_prediction_preview(
    args.output_dir / 'predictions_test.jsonl',
    split_name='test',
    limit=RAW_COMPLETION_PREVIEW_COUNT,
)

manifest = qdora_train_local.build_run_manifest(
    args=args,
    train_records=train_records,
    val_records=val_records,
    test_records=test_records,
    train_examples=train_examples,
    val_examples=val_examples,
    test_examples=test_examples,
    filter_stats={
        'train': train_filter_stats,
        'val': val_filter_stats,
        'test': test_filter_stats,
    },
    model_summary=model_summary,
    train_metrics=train_metrics,
    eval_metrics={'val': val_metrics, 'test': test_metrics},
)
qdora_train_local.write_json(args.output_dir / 'run_manifest.json', manifest)
qdora_train_local.write_json(args.output_dir / 'train_metrics.json', train_metrics)

print('output_dir        =', args.output_dir)
print('train_global_step =', train_metrics.get('global_step'))
print('val_label_match   =', val_metrics.get('label_match'))
print('test_label_match  =', test_metrics.get('label_match'))
print('test_macro_f1     =', test_metrics.get('macro_f1'))

print('output_dir         =', args.output_dir)
print('train_examples     =', manifest['dataset']['kept_examples']['train'])
print('test_examples      =', manifest['dataset']['kept_examples']['test'])
print('train_global_step  =', train_metrics.get('global_step'))
print('latest_checkpoint  =', train_metrics.get('latest_trainer_checkpoint'))
print('test_label_match   =', test_metrics.get('label_match'))
print('test_macro_f1      =', test_metrics.get('macro_f1'))
print('test_format_strict =', test_metrics.get('format_strict_ok'))
print('test_parse_ok      =', test_metrics.get('parse_ok'))
print('test_calibration   =', test_metrics.get('calibration'))
