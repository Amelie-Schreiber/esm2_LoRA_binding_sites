import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import wandb
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score
from datetime import datetime


# Constants
MODEL_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"
MAX_SEQUENCE_LENGTH = 1024
ID2LABEL = {
    0: "No binding site",
    1: "Binding site"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Utility functions
def convert_binding_string_to_labels(binding_string):
    return [1 if char == '+' else 0 for char in binding_string]

def truncate_labels(labels, max_length):
    return [label[:max_length] for label in labels]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]
    true_labels = [
        l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]
    return {"accuracy": accuracy_score(true_labels, true_predictions)}

def run_training():
    # Initialize wandb
    run = wandb.init()

    # Retrieve hyperparameters from wandb
    lr = run.config.learning_rate
    batch_size = run.config.batch_size
    num_epochs = run.config.num_epochs
    weight_decay = run.config.weight_decay

    # Parse the XML file
    tree = ET.parse('binding_sites.xml')
    root = tree.getroot()

    # Extract all sequences and labels
    all_sequences = [partner.find(".//proSeq").text for partner in root.findall(".//BindPartner")]
    all_labels = [convert_binding_string_to_labels(partner.find(".//proBnd").text) for partner in root.findall(".//BindPartner")]

    # Split the dataset into train and test sets
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(all_sequences, all_labels, test_size=0.25, shuffle=True)

    # Tokenize the sequences
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt", is_split_into_words=False)
    test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt", is_split_into_words=False)

    train_labels = truncate_labels(train_labels, MAX_SEQUENCE_LENGTH)
    test_labels = truncate_labels(test_labels, MAX_SEQUENCE_LENGTH)

    # Convert the tokenized data into datasets
    train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
    test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

    # Initialize model and token classification head
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Define the LoraConfig
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=16, 
        lora_alpha=16, 
        target_modules=["query", "key", "value"],
        lora_dropout=0.1, 
        bias="all"
    )

    # Convert the model into a PeftModel
    model = get_peft_model(model, peft_config)

    # Training setup
    training_args = TrainingArguments(
        output_dir=wandb.run.dir,  # Save within the wandb directory; makes it easier to sync
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",   # Enable logging to wandb
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()
    print(f"Best model saved at: {training_args.output_dir}")

    best_model_dir = os.path.join(training_args.output_dir, sorted(os.listdir(training_args.output_dir))[-1])
    print(f"Best model directory: {best_model_dir}")

def train_with_best_hyperparameters(best_hyperparameters):
    """
    Train the model with the best hyperparameters obtained from the sweep
    """
    # Retrieve hyperparameters from the best run
    lr = best_hyperparameters['learning_rate']
    batch_size = best_hyperparameters['batch_size']
    num_epochs = best_hyperparameters['num_epochs']
    weight_decay = best_hyperparameters['weight_decay']

    # Parse the XML file
    tree = ET.parse('binding_sites.xml')
    root = tree.getroot()

    # Extract all sequences and labels
    all_sequences = [partner.find(".//proSeq").text for partner in root.findall(".//BindPartner")]
    all_labels = [convert_binding_string_to_labels(partner.find(".//proBnd").text) for partner in root.findall(".//BindPartner")]

    # Split the dataset into train and test sets
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(all_sequences, all_labels, test_size=0.25, shuffle=True)

    # Tokenize the sequences
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt", is_split_into_words=False)
    test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt", is_split_into_words=False)

    train_labels = truncate_labels(train_labels, MAX_SEQUENCE_LENGTH)
    test_labels = truncate_labels(test_labels, MAX_SEQUENCE_LENGTH)

    # Convert the tokenized data into datasets
    train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
    test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

    # Initialize model and token classification head
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Define the LoraConfig
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=16, 
        lora_alpha=16, 
        target_modules=["query", "key", "value"],
        lora_dropout=0.1, 
        bias="all"
    )

    # Convert the model into a PeftModel
    model = get_peft_model(model, peft_config)

    # Training setup
    training_args = TrainingArguments(
        output_dir="best_model_dir",  # Change this as necessary
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",   # Enable logging to wandb
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()

    # After training, save the model
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join("best_model_dir", f"final_best_model_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Best model saved at: {save_path}")


def main():
    # Define the sweep configuration
    sweep_config = {
        "name": "lora-binding-sites-sweep",
        "method": "bayes",
        "metric": {
            "goal": "minimize",
            "name": "eval_loss"
        },
        "parameters": {
            "learning_rate": {
                "distribution": "uniform",
                "min": 1e-5,
                "max": 1e-3
            },
            "weight_decay": {
                "values": [0.01, 0.03, 0.05]
            },
            "batch_size": {
                "values": [2, 4, 8]
            },
            "num_epochs": {
                "values": [3, 5, 7, 10]
            }
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="lora-binding-sites-predictor")

    # Start the sweep agent
    wandb.agent(sweep_id, function=run_training, count=10)  # Running 10 times

    # After the sweep, retrieve the best run's hyperparameters
    api = wandb.Api()
    sweep = api.sweep(f"amelie-schreiber-math/lora-binding-sites-predictor/{sweep_id}")
    
    # Insert the print statement here
    print(sweep.runs[0].summary_metrics)

    runs_with_eval_loss = [run for run in sweep.runs if 'eval/loss' in run.summary_metrics]
    if runs_with_eval_loss:
        best_run = sorted(runs_with_eval_loss, key=lambda run: run.summary_metrics['eval/loss'])[0]
    else:
        raise ValueError("No runs found with 'eval/loss' metric.")

    best_hyperparameters = best_run.config

    # Train with the best hyperparameters
    train_with_best_hyperparameters(best_hyperparameters)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Setting the GPU to be used
    main()
