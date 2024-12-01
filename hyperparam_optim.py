import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
# from one_step_tot import generate_prompt
from optuna import create_study
# from optuna.integration import TransformersTrainerCallback
import optuna
import json

model_name = "HuggingFaceTB/SmolLM-360M"
save_path = "./smollm_finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Check and add pad token
print("Pad token:", tokenizer.pad_token)
print("Pad token ID:", tokenizer.pad_token_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the dataset
"""
Assume new finetune.csv dataset in following format:
Rank, Puzzle, Response
"2","1 1 11 11","1 + 11 = 12 (left: 1 11 12)\n1 + 11 = 12 (left: 12 12)\n12 + 12 = 24 (left: 24)\nAnswer: (1 + 11) + (1 + 11) = 24"
"""
dataset_path = "./datasets/finetune.csv"
dataset = load_dataset("csv", data_files=dataset_path)

# Split the dataset into 90% train, 10% validation
train_test_split = dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


def generate_prompt(puzzle):
    """One Step Tree-of-Thoughts prompting."""
    # 5 examples
    in_context_demo = '''
        Input: 4 4 6 8
        Steps:
        4 + 8 = 12 (left: 4 6 12)
        6 - 4 = 2 (left: 2 12)
        2 * 12 = 24 (left: 24)
        Answer: (6 - 4) * (4 + 8) = 24\n
        Input: 2 9 10 12
        Steps:
        12 * 2 = 24 (left: 9 10 24)
        10 - 9 = 1 (left: 1 24)
        24 * 1 = 24 (left: 24)
        Answer: (12 * 2) * (10 - 9) = 24\n
        Input: 4 9 10 13
        Steps:
        13 - 10 = 3 (left: 3 4 9)
        9 - 3 = 6 (left: 4 6)
        4 * 6 = 24 (left: 24)
        Answer: 4 * (9 - (13 - 10)) = 24\n
        Input: 1 4 8 8
        Steps:
        8 / 4 = 2 (left: 1 2 8)
        1 + 2 = 3 (left: 3 8)
        3 * 8 = 24 (left: 24)
        Answer: (1 + 8 / 4) * 8 = 24\n
        Input: 5 5 5 9
        Steps:
        5 + 5 = 10 (left: 5 9 10)
        10 + 5 = 15 (left: 9 15)
        15 + 9 = 24 (left: 24)
        Answer: ((5 + 5) + 5) + 9 = 24\n
    '''

    system_prompt = (
        "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.\n"
        "Step 1: Start by considering possible operations for each pair of numbers.\n"
        "Step 2: Try a path (a pair of two numbers), see if the remaining numbers can possibly reach the goal 24. If not, backtrack and attempt another.\n"
        "Step 3: Branch out to try different orders of operations and combinations, evaluating each outcome.\n"
        "Step 4: If one path doesn't lead to a solution, backtrack and try alternative operations.\n"
    )

    prompt = (
        f"{system_prompt}"
        f"{in_context_demo}"
        f"Now, solve the following puzzle:\n{puzzle}\n"
        "Output in the same format as this example including three steps and final answer:\n"
        "Steps:\n"
        "5 + 5 = 10 (left: 5 9 10)\n"
        "10 + 5 = 15 (left: 9 15)\n"
        "15 + 9 = 24 (left: 24)\n"
        "Answer: ((5 + 5) + 5) + 9 = 24"
    )
    return prompt


def preprocess_function(examples):
    """Tokenize the inputs and set the answer as the target label."""

    inputs = [
        f"<|im_start|>user\n{generate_prompt(question)}<|im_end|>\n<|im_start|>assistant"
        for question in examples["Puzzle"]
    ]
    outputs = [f"{answer}<|im_end|>" for answer in examples["Response"]]

    # Tokenize inputs and outputs with consistent padding and truncation
    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        padding="max_length",  # Use "max_length" for consistent tensor size
        truncation=True
    )
    labels = tokenizer(
        outputs,
        max_length=1024,
        padding="max_length",  # Use "max_length" for consistent tensor size
        truncation=True
    )["input_ids"]

    # Replace padding token ids in labels with -100 to ignore in loss computation
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

tokenized_train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names
)

# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,  # Use the model to ensure compatibility
#     padding=True,  # Dynamically pad inputs and labels
#     max_length=1024,  # Truncate if needed
#     return_tensors="pt"  # Return PyTorch tensors
# )

# Define the objective function for hyperparameter tuning
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./smollm_finetune_test",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_train_epochs,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to=None,  # Disable reporting for cleaner Optuna integration
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        # data_collator=data_collator,
    )

    #clear cache
    torch.cuda.empty_cache()

    # Train the model
    trainer.train()

    # Evaluate the model and return the evaluation metric for optimization
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

# Create an Optuna study
study = create_study(direction="minimize")  # Minimize eval_loss
study.optimize(objective, n_trials=20)  # Adjust n_trials based on resources

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

# Save the fine-tuned model with the best parameters
best_params = study.best_params
training_args = TrainingArguments(
    output_dir="./SmolLM_360M_finetuned_best",
    per_device_train_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    num_train_epochs=best_params["num_train_epochs"],
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=True,
    logging_dir="./train_logs",  # Directory to save logs
    report_to="all",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)
trainer.train()
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

log_history = trainer.state.log_history  # Contains training and eval logs

# Save logs to a JSON file
with open("training_logs.json", "w") as f:
    json.dump(log_history, f)