# Import necessary libraries
import pandas as pd
import torch
import os
import shutil
import wandb  # Add wandb import
from datasets import Dataset  # Used for creating Hugging Face Dataset objects
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Changed KFold to StratifiedKFold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)
import numpy as np  # For calculating mean and std of metrics
from datetime import datetime

# Enable wandb for tracking training progress across folds
os.environ["WANDB_DISABLED"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set random seeds for reproducibility
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def check_disk_space(path, min_gb=5):
    """Check if there's enough free disk space"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        print(f"Free disk space: {free_gb:.2f} GB")
        if free_gb < min_gb:
            print(f"⚠️  Warning: Low disk space! Only {free_gb:.2f} GB free")
            return False
        return True
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def cleanup_checkpoints(output_dir):
    """Clean up intermediate checkpoints to save space"""
    try:
        checkpoint_dirs = [
            d for d in os.listdir(output_dir) if d.startswith("checkpoint-")
        ]
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = os.path.join(output_dir, checkpoint_dir)
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                print(f"Cleaned up checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Could not clean up checkpoints: {e}")


# --- Configuration ---
# Define the path to your CSV file
# IMPORTANT: Replace 'your_data.csv' with the actual path to your CSV file.
# Ensure your CSV has 'text' and 'labels' columns.
name = "cleaned"

CSV_FILE_PATH = f"data/covidbr_labeled_{name}.csv"
# CSV_FILE_PATH = "data/test.csv"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_filename = f"qwen3_{name}_{timestamp}.csv"

MODEL_NAME = (
    "Qwen/Qwen3-0.6B"  # Or "Qwen/Qwen-1_8B" if you prefer a slightly larger model
)
OUTPUT_DIR = "./qwen3-0.6_finetuned_model_{name}"
# REDUCED BATCH_SIZE to conserve GPU memory. You can try 4 if 2 is too slow.
BATCH_SIZE = 1  # fixed 1 to qwen
NUM_EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 4e-5  # Standard learning rate for fine-tuning transformers
# Set a reasonable max_length for tokenization to prevent excessive memory usage from very long texts.
# Common values are 128, 256, 512. Adjust based on your typical text length.
MAX_SEQ_LENGTH = 1024
# Gradient accumulation allows simulating a larger batch size without increasing memory per step.
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 4
# Number of folds for K-Fold Cross-Validation
N_SPLITS = 5

# Metrics to track during evaluation
METRICS = ["accuracy", "precision", "recall", "f1-macro", "f1-micro", "fpr"]

# Disk space management
SAVE_ONLY_BEST = True  # Only save the best model to conserve disk space
MIN_DISK_SPACE_GB = 10  # Minimum disk space required before training

# --- 1. Load and Preprocess Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    # Ensure the required columns exist
    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'labels' columns.")
    print("Data loaded successfully.")
    print(f"Initial data shape: {df.shape}")
    print("Sample data:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    print("Please make sure the CSV_FILE_PATH variable points to your actual dataset.")
    exit()
except ValueError as e:
    print(f"Data loading error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- 2. Load Tokenizer ---
print(f"\nLoading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Qwen models typically require adding a pad token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
print("Tokenizer loaded successfully.")


# --- 3. Tokenize Data (Full Dataset) ---
# Function to tokenize the text
def tokenize_function(examples):
    # Explicitly setting max_length here to prevent OOM errors from very long sequences
    # `truncation=True` will cut texts longer than max_length
    # `padding="max_length"` will pad shorter texts to max_length
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )


print(f"\nTokenizing full dataset with max_length={MAX_SEQ_LENGTH}...")
# Convert full DataFrame to Hugging Face Dataset for tokenization
full_dataset = Dataset.from_pandas(df)
tokenized_full_dataset = full_dataset.map(tokenize_function, batched=True)
# Remove original 'text' and '__index_level_0__' columns as they are no longer needed
# Updated to remove 'cleanLinks' based on the error message, as it's an auxiliary column.
tokenized_full_dataset = tokenized_full_dataset.remove_columns(["text", "cleanLinks"])
# Set the format of the datasets to PyTorch tensors
tokenized_full_dataset.set_format("torch")
print("Full dataset tokenized and formatted for PyTorch.")


# --- 4. Define Compute Metrics Function ---
# This function computes comprehensive metrics including accuracy, precision, recall, F1-scores, and FPR
def compute_metrics(p):
    predictions, labels = p
    # Get the class with the highest probability
    preds = torch.argmax(torch.tensor(predictions), axis=1).numpy()

    # Calculate basic metrics
    precision, recall, f1_binary, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    # Calculate macro and micro F1 scores
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    # Calculate False Positive Rate (FPR) - handle edge cases
    try:
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            # Normal case: both classes present
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        elif cm.shape == (1, 1):
            # Edge case: only one class present in predictions and/or labels
            unique_labels = set(labels) | set(preds)
            if len(unique_labels) == 1:
                # All predictions and labels are the same class
                if list(unique_labels)[0] == 0:
                    # All are class 0 (negative), so FPR = 0
                    fpr = 0.0
                else:
                    # All are class 1 (positive), FPR undefined, set to 0
                    fpr = 0.0
            else:
                fpr = 0.0
        else:
            fpr = 0.0
    except Exception as e:
        print(f"Warning: Could not calculate FPR due to confusion matrix shape: {e}")
        fpr = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1_binary,  # Keep for compatibility
        "f1-macro": f1_macro,
        "f1-micro": f1_micro,
        "fpr": fpr,
    }


# --- 5. Stratified K-Fold Cross-Validation ---
print(f"\nStarting {N_SPLITS}-Fold Stratified Cross-Validation...")

# Changed from KFold to StratifiedKFold and passing labels for stratification
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

all_eval_results = []
fold_models = []  # Store model paths and their performance
best_fold_info = {"fold": 0, "f1": -1, "path": "", "trainer": None}

# Modified kf.split to include labels for stratification
for fold, (train_index, val_index) in enumerate(kf.split(df, df["labels"])):
    print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")

    # Create train and validation datasets for the current fold
    train_dataset_fold = tokenized_full_dataset.select(train_index.tolist())
    val_dataset_fold = tokenized_full_dataset.select(val_index.tolist())

    print(f"Fold {fold+1} Training set size: {len(train_dataset_fold)}")
    print(f"Fold {fold+1} Validation set size: {len(val_dataset_fold)}")

    # Load a fresh model for each fold to ensure independent training
    print(f"Loading fresh model for Fold {fold+1}: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    print("Model loaded.")

    # Define Training Arguments for the current fold
    # Output directory for each fold
    fold_output_dir = f"{OUTPUT_DIR}_fold_{fold+1}"
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs_fold_{fold+1}",
        logging_strategy="epoch",
        eval_strategy="epoch",  # Corrected from 'eval_strategy'
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        fp16=True,
        seed=RANDOM_STATE,  # Set random seed for reproducibility
        # Disk space optimization settings
        save_total_limit=1,  # Keep only the best checkpoint to save disk space
        save_only_model=True,  # Don't save optimizer state to reduce file size
        dataloader_pin_memory=False,  # Reduce memory usage
        remove_unused_columns=True,  # Remove unused data columns
        # Wandb configuration for real-time tracking
        report_to="wandb",
        run_name=f"qwen3-covid19-fold-{fold+1}-{timestamp}",
    )
    print("Training arguments defined for fold.")

    # Initialize wandb for this fold with proper grouping

    # Initialize wandb for this specific fold
    wandb.init(
        project="qwen3-covid19-misinfo",
        name=f"{name}-fold-{fold+1}",
        group=f"cv-experiment-{timestamp}",  # Group all folds together
        job_type="fold-training",
        config={
            "fold": fold + 1,
            "total_folds": N_SPLITS,
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "random_state": RANDOM_STATE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "dataset": CSV_FILE_PATH,
            "timestamp": timestamp,
            "train_size": len(train_dataset_fold),
            "val_size": len(val_dataset_fold),
        },
        tags=[
            "cross-validation",
            "qwen3",
            "covid19",
            "misinformation",
            f"fold-{fold+1}",
        ],
        reinit=True,  # Allow reinitializing for each fold
    )

    # Initialize Trainer for the current fold
    print("Initializing Trainer for fold...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_fold,
        eval_dataset=val_dataset_fold,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    print("Trainer initialized for fold.")

    # Check disk space before training
    if not check_disk_space(".", min_gb=10):
        print(
            "Insufficient disk space for training. Please free up space and try again."
        )
        exit()

    # Train the model for the current fold
    print(f"Starting model training for Fold {fold+1}...")
    try:
        trainer.train()
        print(f"Training completed for Fold {fold+1}!")
    except RuntimeError as e:
        if "file write failed" in str(e) or "disk" in str(e).lower():
            print(f"⚠️  Disk space error during training: {e}")
            print("Attempting to clean up and continue...")
            cleanup_checkpoints(fold_output_dir)
            # Try to save just the model without optimizer state
            try:
                trainer.save_model(fold_output_dir)
                print("Model saved successfully after cleanup")
            except Exception as save_error:
                print(f"Could not save model after cleanup: {save_error}")
                continue  # Skip to next fold
        else:
            raise e  # Re-raise if not a disk space issue

    # Evaluate the model for the current fold
    print(f"Evaluating the fine-tuned model on validation set for Fold {fold+1}...")
    eval_results = trainer.evaluate()
    print(f"Fold {fold+1} Evaluation Results: {eval_results}")
    all_eval_results.append(eval_results)

    # Log final fold results to wandb
    fold_summary = {}
    for metric, value in eval_results.items():
        clean_metric = metric.replace("eval_", "")
        fold_summary[f"final_{clean_metric}"] = value

    fold_summary["fold_number"] = fold + 1
    fold_summary["is_best_fold"] = False  # Will update later if this becomes best
    wandb.log(fold_summary)

    # Finish this fold's wandb run
    wandb.finish()

    # Track the best performing fold based on F1 score
    current_f1 = eval_results.get("eval_f1", 0.0)
    if current_f1 > best_fold_info["f1"]:
        # Clean up previous best model if it exists
        if best_fold_info["path"] and os.path.exists(best_fold_info["path"]):
            print(f"Cleaning up previous best model at {best_fold_info['path']}")
            shutil.rmtree(best_fold_info["path"])

        best_fold_info.update(
            {
                "fold": fold + 1,
                "f1": current_f1,
                "path": fold_output_dir,
                "trainer": trainer,
            }
        )
        print(f"🏆 New best model! Fold {fold+1} with F1 score: {current_f1:.4f}")

        if SAVE_ONLY_BEST:
            # Save only the best model so far
            print(f"Saving best model for Fold {fold+1} to {fold_output_dir}...")
            try:
                if check_disk_space(".", min_gb=5):
                    trainer.save_model(fold_output_dir)
                    tokenizer.save_pretrained(fold_output_dir)
                    print("Best model and tokenizer saved.")
                    cleanup_checkpoints(fold_output_dir)
                else:
                    print(
                        f"⚠️  Insufficient disk space to save best model for Fold {fold+1}"
                    )
            except Exception as e:
                print(f"Error saving best model for Fold {fold+1}: {e}")
    else:
        print(
            f"Fold {fold+1} F1 score {current_f1:.4f} did not exceed best: {best_fold_info['f1']:.4f}"
        )
        if not SAVE_ONLY_BEST:
            # Save all models (original behavior)
            print(f"Saving model for Fold {fold+1} to {fold_output_dir}...")
            try:
                if check_disk_space(".", min_gb=5):
                    trainer.save_model(fold_output_dir)
                    tokenizer.save_pretrained(fold_output_dir)
                    print("Model and tokenizer saved for fold.")
                    cleanup_checkpoints(fold_output_dir)
                else:
                    print(f"⚠️  Insufficient disk space to save model for Fold {fold+1}")
                    print("Skipping model save to conserve space")
            except Exception as e:
                print(f"Error saving model for Fold {fold+1}: {e}")
                print("Training results will still be recorded")

# --- 6. Aggregate Results ---
print("\n--- Aggregated Cross-Validation Results ---")
avg_metrics = {
    metric: np.mean([res[metric] for res in all_eval_results if metric in res])
    for metric in all_eval_results[0].keys()
}
std_metrics = {
    metric: np.std([res[metric] for res in all_eval_results if metric in res])
    for metric in all_eval_results[0].keys()
}

print("Average Metrics across all folds:")
for metric, value in avg_metrics.items():
    print(f"  {metric}: {value:.4f} (Std Dev: {std_metrics.get(metric, 0):.4f})")

# --- 6.0. Initialize wandb for cross-validation summary ---
wandb.init(
    project="qwen3-covid19-misinfo",
    name=f"cv-summary-{timestamp}",
    group=f"cv-experiment-{timestamp}",  # Same group as individual folds
    job_type="cv-summary",
    config={
        "experiment_type": "cross_validation_summary",
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "save_only_best": SAVE_ONLY_BEST,
        "dataset": CSV_FILE_PATH,
        "timestamp": timestamp,
    },
    tags=["cross-validation", "qwen3", "covid19", "misinformation", "summary"],
    reinit=True,
)

# Log aggregated metrics
wandb_metrics = {}
for metric, value in avg_metrics.items():
    clean_metric = metric.replace("eval_", "")
    wandb_metrics[f"cv_avg_{clean_metric}"] = value
    wandb_metrics[f"cv_std_{clean_metric}"] = std_metrics.get(metric, 0)

# Log individual fold results
for fold_idx, fold_results in enumerate(all_eval_results):
    for metric, value in fold_results.items():
        clean_metric = metric.replace("eval_", "")
        wandb_metrics[f"fold_{fold_idx+1}_{clean_metric}"] = value

# Log best model info
if best_fold_info["fold"] > 0:
    wandb_metrics["best_fold"] = best_fold_info["fold"]
    wandb_metrics["best_f1"] = best_fold_info["f1"]

wandb.log(wandb_metrics)

# Create a summary table for wandb (individual folds only)
fold_table_data = []
for fold_idx, fold_results in enumerate(all_eval_results):
    row = [fold_idx + 1]
    for metric in [
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_f1-macro",
        "eval_f1-micro",
        "eval_fpr",
    ]:
        if metric in fold_results:
            row.append(round(float(fold_results[metric]), 4))
        else:
            row.append(0.0)
    fold_table_data.append(row)

# Create wandb table for individual folds
fold_table = wandb.Table(
    columns=[
        "Fold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "F1-Macro",
        "F1-Micro",
        "FPR",
    ],
    data=fold_table_data,
)
wandb.log({"fold_results_table": fold_table})

# Create a separate summary table for averages and std devs
summary_table_data = []
# Add average row
avg_row = ["Average"]
for metric in [
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "eval_f1",
    "eval_f1-macro",
    "eval_f1-micro",
    "eval_fpr",
]:
    if metric in avg_metrics:
        avg_row.append(round(float(avg_metrics[metric]), 4))
    else:
        avg_row.append(0.0)
summary_table_data.append(avg_row)

# Add std dev row
std_row = ["Std Dev"]
for metric in [
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "eval_f1",
    "eval_f1-macro",
    "eval_f1-micro",
    "eval_fpr",
]:
    if metric in std_metrics:
        std_row.append(round(float(std_metrics[metric]), 4))
    else:
        std_row.append(0.0)
summary_table_data.append(std_row)

# Create wandb table for summary statistics
summary_table = wandb.Table(
    columns=[
        "Statistic",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "F1-Macro",
        "F1-Micro",
        "FPR",
    ],
    data=summary_table_data,
)
wandb.log({"cv_summary_table": summary_table})

# Log best model metrics as a separate section
if best_fold_info["fold"] > 0:
    best_fold_results = all_eval_results[best_fold_info["fold"] - 1]
    best_model_metrics = {}
    for metric, value in best_fold_results.items():
        clean_metric = metric.replace("eval_", "")
        best_model_metrics[f"best_model_{clean_metric}"] = value
    wandb.log(best_model_metrics)

print(
    f"📊 Cross-validation summary logged to wandb: qwen3-covid19-misinfo/cv-summary-{timestamp}"
)
print(f"📊 Individual fold training tracked in group: cv-experiment-{timestamp}")

# --- 6.1. Save Metrics to CSV ---


# Prepare detailed fold metrics for CSV
fold_metrics_data = []
for fold_idx, fold_results in enumerate(all_eval_results):
    fold_data = {
        "fold": fold_idx + 1,
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
    }

    # Add all evaluation metrics
    for metric, value in fold_results.items():
        # Remove 'eval_' prefix if present
        clean_metric = metric.replace("eval_", "")
        fold_data[clean_metric] = value

    fold_metrics_data.append(fold_data)

# Add summary row with averages and std deviations
summary_data = {
    "fold": "AVERAGE",
    "timestamp": timestamp,
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "max_seq_length": MAX_SEQ_LENGTH,
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
}

for metric, value in avg_metrics.items():
    clean_metric = metric.replace("eval_", "")
    summary_data[clean_metric] = value

fold_metrics_data.append(summary_data)

# Add std deviation row
std_data = {
    "fold": "STD_DEV",
    "timestamp": timestamp,
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "max_seq_length": MAX_SEQ_LENGTH,
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
}

for metric, value in std_metrics.items():
    clean_metric = metric.replace("eval_", "")
    std_data[clean_metric] = value

fold_metrics_data.append(std_data)

# Save to CSV
metrics_df = pd.DataFrame(fold_metrics_data)
try:
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"📊 Fold metrics saved to: {metrics_filename}")
    print(f"   Columns: {list(metrics_df.columns)}")
    print(f"   Rows: {len(fold_metrics_data)} (including average and std dev)")
except Exception as e:
    print(f"⚠️  Error saving metrics to CSV: {e}")

# --- 7. Best Model Summary and Final Save ---
if best_fold_info["fold"] > 0:
    print(f"\n🏆 BEST MODEL SUMMARY:")
    print(f"  Best Fold: {best_fold_info['fold']}")
    print(f"  Best F1 Score: {best_fold_info['f1']:.4f}")

    # Get the actual best fold metrics from all_eval_results
    best_fold_results = all_eval_results[best_fold_info["fold"] - 1]
    print(f"  Best Fold Metrics:")
    for metric, value in best_fold_results.items():
        clean_metric = metric.replace("eval_", "")
        print(f"    {clean_metric}: {value:.4f}")
    print(f"  Model saved at: {best_fold_info['path']}")

    if SAVE_ONLY_BEST:
        # Copy the best model to the main output directory
        final_output_path = OUTPUT_DIR
        if best_fold_info["path"] != final_output_path:
            try:
                if os.path.exists(final_output_path):
                    shutil.rmtree(final_output_path)
                shutil.copytree(best_fold_info["path"], final_output_path)
                print(f"✅ Best model copied to final location: {final_output_path}")
            except Exception as e:
                print(f"Error copying best model to final location: {e}")

        print(f"\n💾 DISK SPACE SAVINGS:")
        print(
            f"  Only the best performing model (Fold {best_fold_info['fold']}) was saved"
        )
        print(f"  This saved approximately {(N_SPLITS - 1) * 1.2:.1f} GB of disk space")
    else:
        print(f"\n💾 All {N_SPLITS} fold models were saved to disk")

# Finish wandb run
wandb.finish()

print("\n📈 RESULTS SUMMARY:")
print(f"  Cross-validation metrics saved to: {metrics_filename}")
print(f"  Experiment logged to wandb: qwen3-covid19-misinfo")
print(
    f"  Best model available at: {OUTPUT_DIR if SAVE_ONLY_BEST else 'individual fold directories'}"
)
print(f"  Run timestamp: {timestamp}")
print("\nYou can now use the CSV file to plot and analyze the fold performance!")
