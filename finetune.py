import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the Orca model and tokenizer
# https://huggingface.co/microsoft/Orca-2-7b
# model_name = "microsoft/Orca-2-7b"
# save_path = "./orca_finetuned"

# Can be replaced with a smaller alternative as discussed.
# Use this currently for testing
model_name = "HuggingFaceTB/SmolLM-360M"
save_path = "./smollm_finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load the dataset
"""
Assume the csv dataset is in the following format:

question,answer
"6 9 12 13", "Steps:\n13 - 9 = 4 (left: 4 6 12)\n 12 / 4 = 3 (left: 3 6)\n 3 * 6 = 18 (left: 18)\n18 + 6 = 24 (left: 24)\nAnswer: ((13 - 9) / 4) * 6 + 6 = 24"

"""
dataset_path = "./datasets/finetune.csv"
dataset = load_dataset("csv", data_files=dataset_path)

# Split the dataset into 90% train, 10% validation
train_test_split = dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

def preprocess_function(examples):
    """Tokenize the inputs and set the answer as the target label."""
    inputs = [
        f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"
        for question in examples["Puzzle"]
    ]
    outputs = [f"{answer}<|im_end|>" for answer in examples["Response"]]
    model_inputs = tokenizer(inputs, text_target=outputs,
                             max_length=1024, truncation=True)
    return model_inputs


tokenized_train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names
)

# TODO: Adjust hyperparameters
training_args = TrainingArguments(
    output_dir="./orca_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,  # Pass in the hyperparameters
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
