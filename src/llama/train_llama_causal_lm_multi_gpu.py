import sys
sys.path.append("../../util/")

import json
import os
import pandas as pd
import pers_labels
import torch
import warnings

from accelerate import Accelerator
from datasets import Dataset
from datetime import datetime
from peft import LoraConfig
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MAX_SEQ_LENGTH = 8192

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DATA_TYPE = "mbpt_0_top_lbl"

LABEL_MODE = pers_labels.MBPT if pers_labels.MBPT.lower() in DATA_TYPE else pers_labels.BIG_5

pers_defs = {
    pers_labels.MBPT: {
        0: (("I", "introverted"), ("E", "extroverted")),
        1: (("S", "sensing"), ("N", "intuitive")),
        2: (("F", "feeling"), ("T", "thinking")),
        3: (("J", "judging"), ("P", "perceiving")),
    },
    pers_labels.BIG_5: {
        0: (("S", "social"), ("R", "reserved")),
        1: (("L", "limbic"), ("C", "calm")),
        2: (("O", "organized"), ("U", "unstructured")),
        3: (("A", "agreeable"), ("E", "egocentric")),
        4: (("N", "non-curious"), ("I", "inquisitive")),
    }
}

index = [idx for idx in range(5) if str(idx) in DATA_TYPE][0]

((label1, label1_def), (label2, label2_def)) = pers_defs[LABEL_MODE][index]


def create_prompt(text, char):
        _, scene = text.split("\n", 1)

        return \
    f"""Read the scenes below and then categorize {char}'s personality as "{label1}" for {label1_def} or "{label2}" for {label2_def}, according to the {LABEL_MODE.lower()} personality typology. Response with only one word.

    scenes:
    {scene}

    """

def to_messages(df):
    messages = []
    df.to_dict(orient="records")
    for row in df.iloc:
        messages.append(([{
            "role": "user",
            "content": row.text
        }, {
            "role": "assistant",
            "content": row.label
        }]))
    return messages

def main(train_df, valid_df):
    # SETTING UP BITS_AND_BYTES

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # ==========================================================================
    # SETTING UP TOKENIZER

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tkr = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/nlp/scr/neigbe/.cache",
        model_max_length=MAX_SEQ_LENGTH,
        pad_token="<|pad_id|>"
    )
    # ==========================================================================
    # SETTING UP MODEL

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map,
        attn_implementation = "flash_attention_2",
        cache_dir="/nlp/scr/neigbe/.cache")

    model.resize_token_embeddings(len(tkr), pad_to_multiple_of=8)
    model.config.pad_token_id = tkr.pad_token_id

    # ==========================================================================
    # SETTING UP LORA

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        bias= "none",
        task_type= "CAUSAL_LM",
        lora_dropout=0.1,
        inference_mode= False,
    )

    # ==========================================================================
    # LOADING DATASET

    train_messages = to_messages(train_df)

    train_messages_fmt = []
    for msg in train_messages:
        fmt_msg = tkr.apply_chat_template(msg, tokenize=False)
        train_messages_fmt.append(fmt_msg)
    final_train_df = pd.DataFrame(train_messages_fmt, columns=["text"])

    train = Dataset.from_pandas(final_train_df, split="train").with_format("torch")

    valid_messages = to_messages(valid_df)

    valid_messages_fmt = []
    for msg in valid_messages:
        fmt_msg = tkr.apply_chat_template(msg, tokenize=False)
        valid_messages_fmt.append(fmt_msg)
    final_valid_df = pd.DataFrame(valid_messages_fmt, columns=["text"])

    valid = Dataset.from_pandas(final_valid_df, split="valid").with_format("torch")

    # ==========================================================================
    # SETTING UP CONVO FORMAT

    instruction_template="<|start_header_id|>user<|end_header_id|>\n\n"
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tkr)

    # ==========================================================================
    # SETTING UP TRAIN PARAMS

    batch_size = 2

    dt = str(datetime.now()).replace(" ", "|").split(".")[0]
    model_path = f"../models/{DATA_TYPE}/{dt}/"

    training_args = TrainingArguments(output_dir=model_path,
                                    evaluation_strategy="epoch",
                                    logging_strategy="epoch",
                                    save_strategy="epoch",
                                    num_train_epochs=3,
                                    save_total_limit = 5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    disable_tqdm=False,
                                    # torch_compile=True,
                                    gradient_accumulation_steps=2,
                                    gradient_checkpointing=True,
                                    lr_scheduler_type="cosine",
                                    gradient_checkpointing_kwargs={"use_reentrant": True},
                                    load_best_model_at_end=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        peft_config=lora_config,
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=collator,
        packing=False,
        dataset_text_field="text",
    )

    # ==========================================================================
    # STARTING TRAINING

    # warnings.filterwarnings('ignore')

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":

    # LOADING DATASET

    with open("../../data/cornell_movies/speakers.json", "r+") as fp:
        fp_parsed = json.load(fp)
        chars_meta = {}
        chars_meta_rows = []
        for char in fp_parsed:
            meta = fp_parsed[char]["meta"]
            meta["character_name"] = meta["character_name"].lower()
            meta["char_id"] = char
            chars_meta[char] = meta
            chars_meta_rows.append(meta)

    dataset = pd.read_json(f"../data/datasets/{DATA_TYPE}.jsonl", lines=True)

    # SETTING UP PROMPTS

    prompts = []
    for row in dataset.iloc:
        char_name = chars_meta[row.char_id]["character_name"]
        scene = row.text
        prompts.append(create_prompt(scene, char_name))

    dataset["text"] = prompts

    # GETTING TRAIN DATA
    splitter1 = GroupShuffleSplit(test_size=.95, random_state=12)
    non_test_idx, _ = next(splitter1.split(X=dataset[["text"]], y=dataset["label"], groups=dataset["movie_id"]))
    train_df = dataset[["text", "label"]].iloc[non_test_idx]
    non_test_df = dataset.iloc[non_test_idx]

    splitter2 = GroupShuffleSplit(test_size=.2)
    train_idx, valid_idx = next(splitter2.split(X=non_test_df, y=non_test_df["label"], groups=non_test_df["movie_id"]))
    to_split_df = non_test_df[["text", "label"]]

    train_df = to_split_df.iloc[train_idx]
    valid_df = to_split_df.iloc[valid_idx]

    main(train_df, valid_df)
