import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from optuna import create_study
# from optuna.integration import TransformersTrainerCallback
import optuna

model_name = "HuggingFaceTB/SmolLM-360M"
save_path = "./smollm_hptest"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load the dataset
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
        for question in examples["question"]
    ]
    outputs = [f"{answer}<|im_end|>" for answer in examples["answer"]]
    model_inputs = tokenizer(inputs, text_target=outputs,
                             max_length=1024, truncation=True)
    return model_inputs

tokenized_train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names
)

# Define the objective function for hyperparameter tuning
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./orca_finetuned",
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
    )

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
# best_params = study.best_params
# training_args = TrainingArguments(
#     output_dir="./orca_finetuned_best",
#     per_device_train_batch_size=best_params["batch_size"],
#     learning_rate=best_params["learning_rate"],
#     weight_decay=best_params["weight_decay"],
#     num_train_epochs=best_params["num_train_epochs"],
#     save_steps=500,
#     save_total_limit=2,
#     evaluation_strategy="steps",
#     eval_steps=500,
#     logging_steps=100,
#     fp16=True,
#     push_to_hub=False,
#     remove_unused_columns=True,
# )

# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_eval_dataset,
# )
# trainer.train()
# trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)