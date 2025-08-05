# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from: https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py
"""

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from trl import (
    SFTConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    ModelConfig,
    get_peft_config
)
import trl
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig(trl.ModelConfig):
    lora_target_parameters: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of target parameters for LoRA."},
    )


def main(script_args, training_args, model_args):
    # TODO: Remove once models can be pushed to the Hub
    training_args.push_to_hub = False

    # ------------------------
    # Load model & tokenizer
    # ------------------------
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # --------------
    # Load dataset
    # --------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # -------------
    # Train model
    # -------------
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        peft_config.target_parameters = model_args.lora_target_parameters

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft", help="Run the SFT training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)