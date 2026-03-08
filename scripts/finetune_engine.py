"""
Saturday MK1 Fine-Tuning Engine — LoRA/QLoRA Training for Coding Models
======================================================================
Production fine-tuning script supporting Kimi K2.5, Qwen 3.5, GPT-oss 120B,
DeepSeek V3.2, and any HuggingFace-compatible causal LM.

Usage:
    # Dry run (validates config + data loading, no GPU needed)
    python scripts/finetune_engine.py --dry-run --config configs/gpt_oss_120b.yaml

    # Full training run
    python scripts/finetune_engine.py --config configs/qwen3_5.yaml --data-dir training_data/

    # Fine-tune on distilled data only
    python scripts/finetune_engine.py --config configs/gpt_oss_120b.yaml \\
        --data-files distillation_data/distilled_20260307.jsonl

    # Resume training from checkpoint
    python scripts/finetune_engine.py --config configs/gpt_oss_120b.yaml --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saturday-finetune")


# ═══════════════════════════════════════════
# TRAINING CONFIG
# ═══════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Complete configuration for a fine-tuning run."""

    # Model
    base_model: str = "moonshotai/Kimi-K2.5"
    model_type: str = "causal_lm"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_bias: str = "none"

    # Quantization (QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Sequence
    max_seq_length: int = 8192
    packing: bool = True  # Pack short sequences together

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 5
    eval_strategy: str = "steps"
    eval_steps: int = 100
    logging_steps: int = 10

    # Output
    output_dir: str = "./saturday_model"
    run_name: str = "saturday_mk1_v1"

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    optim: str = "paged_adamw_8bit"

    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

    # Data
    data_dir: str = "./training_data"
    data_files: list[str] = field(default_factory=list)
    eval_split: float = 0.05
    shuffle_seed: int = 42

    # Misc
    report_to: str = "none"  # "wandb", "tensorboard", or "none"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        """Validate config and return list of warnings."""
        warnings = []

        if self.use_4bit and not self.use_lora:
            warnings.append("4-bit quantization requires LoRA — enabling LoRA")
            self.use_lora = True

        if self.max_seq_length > 16384:
            warnings.append(
                f"max_seq_length={self.max_seq_length} is very large — "
                "may cause OOM on consumer GPUs"
            )

        if self.per_device_train_batch_size * self.gradient_accumulation_steps < 16:
            warnings.append(
                "Effective batch size < 16 — consider increasing for stability"
            )

        vram_estimate = self._estimate_vram()
        warnings.append(f"Estimated VRAM requirement: ~{vram_estimate}GB")

        return warnings

    def _estimate_vram(self) -> int:
        """Rough VRAM estimation in GB."""
        # Very rough heuristic
        model_lower = self.base_model.lower()

        if "120b" in model_lower:
            base_vram = 24 if self.use_4bit else 240
        elif "397b" in model_lower or "400b" in model_lower:
            base_vram = 80 if self.use_4bit else 800
        elif "685b" in model_lower:
            base_vram = 140 if self.use_4bit else 1400
        elif "70b" in model_lower:
            base_vram = 16 if self.use_4bit else 140
        elif "7b" in model_lower or "8b" in model_lower:
            base_vram = 6 if self.use_4bit else 16
        else:
            base_vram = 24  # conservative default

        # Add overhead for gradients, optimizer states, activations
        if self.gradient_checkpointing:
            overhead = base_vram * 0.3
        else:
            overhead = base_vram * 0.8

        return int(base_vram + overhead)


# ═══════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════

class SaturdayDataLoader:
    """
    Loads and prepares training data from multiple sources.
    Supports ChatML JSONL format and legacy MoE format.
    """

    SATURDAY_SYSTEM_PROMPT = (
        "You are Saturday MK1, an enterprise-grade coding AI assistant built for "
        "Fortune 500 companies. You write production-ready, secure, "
        "well-documented code."
    )

    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_dataset(self) -> tuple[Any, Any]:
        """
        Load and prepare training + eval datasets.

        Returns:
            (train_dataset, eval_dataset) — HuggingFace Dataset objects
        """
        try:
            from datasets import Dataset, concatenate_datasets
        except ImportError:
            raise ImportError("pip install datasets")

        all_samples = []

        # Load from data files (distilled data, explicit files)
        for data_file in self.config.data_files:
            samples = self._load_jsonl(data_file)
            all_samples.extend(samples)
            log.info(f"Loaded {len(samples)} samples from {Path(data_file).name}")

        # Load from data directory (MoE training data)
        if self.config.data_dir and Path(self.config.data_dir).exists():
            dir_samples = self._load_from_directory(self.config.data_dir)
            all_samples.extend(dir_samples)

        if not all_samples:
            raise ValueError("No training data found — check data_dir and data_files")

        log.info(f"Total samples loaded: {len(all_samples):,}")

        # Convert to HuggingFace Dataset
        texts = [self._format_chatml(s) for s in all_samples]
        dataset = Dataset.from_dict({"text": texts})

        # Shuffle and split
        dataset = dataset.shuffle(seed=self.config.shuffle_seed)

        if self.config.eval_split > 0:
            split = dataset.train_test_split(
                test_size=self.config.eval_split,
                seed=self.config.shuffle_seed,
            )
            train_ds, eval_ds = split["train"], split["test"]
        else:
            train_ds, eval_ds = dataset, None

        log.info(
            f"Train: {len(train_ds):,} samples"
            + (f" | Eval: {len(eval_ds):,} samples" if eval_ds else "")
        )

        return train_ds, eval_ds

    def _load_jsonl(self, path: str) -> list[dict]:
        """Load samples from a JSONL file."""
        samples = []
        path = Path(path)

        if not path.exists():
            log.warning(f"File not found: {path}")
            return samples

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "messages" in item:
                        samples.append(item)
                    elif "text" in item and "prompt" in item:
                        # Legacy MoE format
                        samples.append(self._convert_legacy(item))
                except json.JSONDecodeError:
                    log.debug(f"Skipping malformed JSON at line {line_num} in {path.name}")

        return samples

    def _load_from_directory(self, data_dir: str) -> list[dict]:
        """Load all JSONL files from training data directory."""
        samples = []
        data_path = Path(data_dir)

        for jsonl_file in sorted(data_path.rglob("*.jsonl")):
            file_samples = self._load_jsonl(str(jsonl_file))
            samples.extend(file_samples)

        if samples:
            log.info(f"Loaded {len(samples):,} samples from {data_dir}")
        return samples

    def _convert_legacy(self, item: dict) -> dict:
        """Convert legacy MoE format to ChatML."""
        return {
            "messages": [
                {"role": "system", "content": self.SATURDAY_SYSTEM_PROMPT},
                {"role": "user", "content": item.get("prompt", "")},
                {"role": "assistant", "content": item.get("text", "")},
            ],
            "metadata": {
                "source": "moe_legacy",
                "quality_score": item.get("quality_score", 0.95),
            },
        }

    def _format_chatml(self, sample: dict) -> str:
        """
        Format a sample as ChatML text for training.

        ChatML format:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
        ...
        <|im_end|>
        <|im_start|>assistant
        ...
        <|im_end|>
        """
        messages = sample.get("messages", [])
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(formatted_parts)


# ═══════════════════════════════════════════
# FINE-TUNING ENGINE
# ═══════════════════════════════════════════

class SaturdayFineTuner:
    """
    Production LoRA/QLoRA fine-tuning engine.

    Supports:
    - QLoRA (4-bit quantization + LoRA)
    - Gradient checkpointing
    - DeepSpeed ZeRO stages
    - Checkpoint resume
    - Mixed precision training
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> bool:
        """
        Initialize model, tokenizer, and LoRA adapters.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            log.error(
                f"Missing dependencies: {e}\n"
                "Install: pip install transformers peft bitsandbytes accelerate trl"
            )
            return False

        log.info(f"Loading base model: {self.config.base_model}")

        # Quantization config
        bnb_config = None
        if self.config.use_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
            log.info("Using 4-bit QLoRA quantization (nf4 + double quant)")

        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.base_model,
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": "auto",
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Prepare for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )

        # Apply LoRA
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias=self.config.lora_bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

            trainable, total = self.model.get_nb_trainable_parameters()
            log.info(
                f"LoRA applied: {trainable:,} trainable / {total:,} total "
                f"({trainable/total*100:.2f}%)"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        log.info(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
        return True

    def train(self, train_dataset, eval_dataset=None) -> dict:
        """
        Run the fine-tuning training loop.

        Args:
            train_dataset: HuggingFace Dataset for training
            eval_dataset: Optional HuggingFace Dataset for evaluation

        Returns:
            Training results dict
        """
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise ImportError("pip install trl>=0.7.0")

        log.info("=" * 60)
        log.info("STARTING FINE-TUNING")
        log.info("=" * 60)
        log.info(f"  Model:          {self.config.base_model}")
        log.info(f"  LoRA rank:      {self.config.lora_r}")
        log.info(f"  Batch size:     {self.config.per_device_train_batch_size}")
        log.info(f"  Grad accum:     {self.config.gradient_accumulation_steps}")
        log.info(f"  Effective BS:   {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        log.info(f"  Epochs:         {self.config.num_train_epochs}")
        log.info(f"  Learning rate:  {self.config.learning_rate}")
        log.info(f"  Max seq length: {self.config.max_seq_length}")
        log.info(f"  Train samples:  {len(train_dataset):,}")
        if eval_dataset:
            log.info(f"  Eval samples:   {len(eval_dataset):,}")
        log.info(f"  Output:         {self.config.output_dir}")
        log.info("=" * 60)

        # SFT Training config
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            run_name=self.config.run_name,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy=self.config.eval_strategy if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            logging_steps=self.config.logging_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=self.config.report_to,
            seed=self.config.seed,
            dataset_text_field="text",
        )

        # DeepSpeed
        if self.config.use_deepspeed and self.config.deepspeed_config:
            sft_config.deepspeed = self.config.deepspeed_config

        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

        # Train
        start_time = time.time()
        results = self.trainer.train(
            resume_from_checkpoint=self._find_checkpoint() if self.config.output_dir else None,
        )
        elapsed = time.time() - start_time

        log.info(f"\n{'='*60}")
        log.info(f"TRAINING COMPLETE — {elapsed/3600:.1f} hours")
        log.info(f"  Train loss: {results.training_loss:.4f}")
        log.info(f"  Steps:      {results.global_step}")
        log.info(f"{'='*60}")

        return {
            "training_loss": results.training_loss,
            "global_step": results.global_step,
            "elapsed_hours": round(elapsed / 3600, 2),
            "model": self.config.base_model,
            "lora_r": self.config.lora_r,
        }

    def save_model(self, merge_adapters: bool = False):
        """
        Save the fine-tuned model.

        Args:
            merge_adapters: If True, merge LoRA weights into base model
        """
        output_dir = Path(self.config.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)

        if merge_adapters and self.config.use_lora:
            log.info("Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(str(output_dir))
            log.info(f"Merged model saved to: {output_dir}")
        else:
            self.model.save_pretrained(str(output_dir))
            log.info(f"Adapter model saved to: {output_dir}")

        self.tokenizer.save_pretrained(str(output_dir))

        # Save training config
        config_path = output_dir / "saturday_training_config.yaml"
        self.config.to_yaml(str(config_path))

        # Save model card
        self._save_model_card(output_dir)

        log.info(f"Model artifacts saved to: {output_dir}")

    def _find_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for resume."""
        output_dir = Path(self.config.output_dir)
        if not output_dir.exists():
            return None

        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )

        if checkpoints:
            latest = str(checkpoints[-1])
            log.info(f"Resuming from checkpoint: {latest}")
            return latest
        return None

    def _save_model_card(self, output_dir: Path):
        """Generate a MODEL_CARD.md for the fine-tuned model."""
        card = f"""# Saturday MK1 — Fine-Tuned Coding Model

## Model Details
- **Base Model**: {self.config.base_model}
- **Fine-tuning Method**: {'QLoRA (4-bit)' if self.config.use_4bit else 'LoRA'}
- **LoRA Rank**: {self.config.lora_r}
- **LoRA Alpha**: {self.config.lora_alpha}
- **Target Modules**: {', '.join(self.config.lora_target_modules)}
- **Training Epochs**: {self.config.num_train_epochs}
- **Learning Rate**: {self.config.learning_rate}
- **Max Sequence Length**: {self.config.max_seq_length}
- **Trained on**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

## Purpose
Enterprise-grade coding AI assistant, fine-tuned to surpass Claude Opus 4.6
in code generation, security analysis, debugging, and architecture design.

## Capabilities
- Security-first code generation
- Enterprise architecture patterns (CQRS, DDD, microservices)
- Advanced debugging and performance optimization
- Compliance-aware development (GDPR, HIPAA, SOC2, PCI-DSS)
- Multi-file reasoning and refactoring

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
```

## License
Inherits license from base model: {self.config.base_model}
"""
        with open(output_dir / "MODEL_CARD.md", "w") as f:
            f.write(card)


# ═══════════════════════════════════════════
# DRY RUN MODE
# ═══════════════════════════════════════════

def dry_run(config: TrainingConfig):
    """
    Validate everything without loading the actual model (no GPU required).
    Tests: config validation, data loading, format conversion.
    """
    log.info("=" * 60)
    log.info("DRY RUN MODE — No GPU required")
    log.info("=" * 60)

    # Validate config
    log.info("\n[1/4] Validating config...")
    warnings = config.validate()
    for w in warnings:
        log.info(f"  ⚠️  {w}")
    log.info("  ✅ Config valid")

    # Test data loading
    log.info("\n[2/4] Testing data loading...")
    loader = SaturdayDataLoader(config)

    try:
        from datasets import Dataset
        has_datasets_lib = True
    except ImportError:
        has_datasets_lib = False
        log.warning("  ⚠️  'datasets' library not installed — skipping dataset load test")

    if has_datasets_lib:
        try:
            train_ds, eval_ds = loader.load_dataset()
            log.info(f"  ✅ Loaded {len(train_ds):,} train samples")
            if eval_ds:
                log.info(f"  ✅ Loaded {len(eval_ds):,} eval samples")

            # Show sample
            if len(train_ds) > 0:
                sample = train_ds[0]["text"]
                preview = sample[:300] + "..." if len(sample) > 300 else sample
                log.info(f"\n  📄 Sample preview:\n{preview}")
        except Exception as e:
            log.error(f"  ❌ Data loading failed: {e}")
            return False

    # Check dependencies
    log.info("\n[3/4] Checking dependencies...")
    deps = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "peft": "PEFT (LoRA)",
        "bitsandbytes": "BitsAndBytes (quantization)",
        "trl": "TRL (SFT Trainer)",
        "accelerate": "Accelerate (multi-GPU)",
        "datasets": "Datasets",
    }

    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            log.info(f"  ✅ {name}")
        except ImportError:
            log.warning(f"  ❌ {name} — pip install {module}")
            missing.append(module)

    # GPU check
    log.info("\n[4/4] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                log.info(f"  ✅ GPU {i}: {name} ({vram:.1f}GB)")

            vram_needed = config._estimate_vram()
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_mem / (1024**3)
                for i in range(gpu_count)
            )
            if total_vram >= vram_needed:
                log.info(f"  ✅ Total VRAM: {total_vram:.1f}GB >= {vram_needed}GB needed")
            else:
                log.warning(
                    f"  ⚠️  Total VRAM: {total_vram:.1f}GB < {vram_needed}GB needed — "
                    "may OOM during training"
                )
        else:
            log.warning("  ⚠️  No CUDA GPU detected — training requires GPU")
    except ImportError:
        log.warning("  ⚠️  PyTorch not installed — cannot check GPU")

    # Summary
    log.info(f"\n{'='*60}")
    if missing:
        log.info(f"DRY RUN INCOMPLETE — Install missing: pip install {' '.join(missing)}")
    else:
        log.info("DRY RUN PASSED — Ready for training! 🚀")
    log.info(f"{'='*60}")

    return len(missing) == 0


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Saturday MK1 Fine-Tuning Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model name/path (overrides config)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Training data directory (overrides config)",
    )
    parser.add_argument(
        "--data-files", nargs="+", default=None,
        help="Specific JSONL files to train on (overrides config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--lora-r", type=int, default=None,
        help="LoRA rank (overrides config)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and dependencies without training",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapters into base model after training",
    )
    parser.add_argument(
        "--save-config", type=str, default=None,
        help="Save the effective config to a YAML file and exit",
    )

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        config = TrainingConfig.from_yaml(args.config)
        log.info(f"Loaded config from: {args.config}")
    else:
        config = TrainingConfig()
        log.info("Using default config")

    # CLI overrides
    if args.base_model:
        config.base_model = args.base_model
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.data_files:
        config.data_files = args.data_files
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.lora_r:
        config.lora_r = args.lora_r

    # Save config mode
    if args.save_config:
        config.to_yaml(args.save_config)
        log.info(f"Config saved to: {args.save_config}")
        return

    # Dry run mode
    if args.dry_run:
        dry_run(config)
        return

    # Full training
    log.info("=" * 60)
    log.info("SATURDAY MK1 FINE-TUNING ENGINE")
    log.info("=" * 60)

    # Validate
    warnings = config.validate()
    for w in warnings:
        log.info(f"⚠️  {w}")

    # Load data
    loader = SaturdayDataLoader(config)
    train_ds, eval_ds = loader.load_dataset()

    # Setup model
    tuner = SaturdayFineTuner(config)
    if not tuner.setup():
        log.error("Model setup failed — aborting")
        sys.exit(1)

    # Train
    results = tuner.train(train_ds, eval_ds)

    # Save
    tuner.save_model(merge_adapters=args.merge)

    # Save results
    results_file = Path(config.output_dir) / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n🎉 Training complete! Model saved to: {config.output_dir}/final")


if __name__ == "__main__":
    main()
