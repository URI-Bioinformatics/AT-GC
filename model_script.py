import os
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from sequences import genome_tuples

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Constants for testing and configuration.
#   Used to easily set different run metrics like Window Stride, and which K-Mer numbers to use.
KMER_SIZES = [8, 9, 10]
MAX_LEN = 512
STRIDE = 256
EPOCHS = 10
MAX_CHUNKS_PER_GENOME = 100


# This function is what allows the exceptionally long sequences to be broken down into the k-mer chunks
# Generates the k-mers, defines the starting point for the sliding window positions, and randomly limits the number of chunks.
def sliding_window_kmers_limited(seq: str, k: int, window: int, stride: int, max_chunks: int = 100) -> List[str]:
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    indices = list(range(0, len(kmers) - window + 1, stride))
    if len(indices) > max_chunks:
        indices = random.sample(indices, max_chunks)
    return [' '.join(kmers[i:i + window]) for i in indices]


# This class is what allows our dataset to be passed as input into the transformer, as it requires a specific format for the data.
#   It is used in conjunction with the HuggingFace "Trainer" (can also be used with a PyTorch Dataloader)
#   Inherits PyTorch's Dataset Class, which makes it compatible with the PyTorch tools.
class KmerWindowDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_len: int = 512):
        self.encodings = tokenizer(sequences, truncation=True, padding='max_length', max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# BERT is a pre-trained transformer available to download off of HuggingFace, which is what we are doing in this block of code.
# We are using the base version of BERT, which is trained on the English language.
def load_base_bert(num_labels: int):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return tokenizer, model


# This is the secret sauce of the transformer's transfer learning. We are cutting off the last 10% of the weights to retrain the model.
#   We do this while maintaining the pre-trained weights of BERT in order to drastically reduce required compute resources and training times.
def truncate_transformer_weights(model, ratio: float = 0.1):
    encoder = model.bert.encoder
    total_layers = len(encoder.layer)
    cutoff = int(total_layers * (1 - ratio))
    encoder.layer = torch.nn.ModuleList(encoder.layer[:cutoff])
    print(f"Truncated to {cutoff} layers out of {total_layers}")
    return model


# This function handles plotting of the model validation accuracies during training for each k-mer configuration.
# Utilizes the trainer object from Hugging Face to access its history.
def plot_accuracy_curve(trainer, k: int):
    log_history = trainer.state.log_history
    val_acc = [entry["eval_accuracy"] for entry in log_history if "eval_accuracy" in entry]
    epochs = list(range(1, len(val_acc) + 1))

    plt.figure()
    plt.plot(epochs, val_acc, marker='o', label=f"{k}-mer")
    plt.title(f"Validation Accuracy Over Epochs ({k}-mer)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plot_path = f"./results_k{k}/val_accuracy_curve_k{k}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved accuracy plot to {plot_path}")


# This function handles all aspects of the model training. This includes stratifying the dataset, chunking and labeling the data, and making
#   directory in which the results will be stored.
def train_model(sequences: List[str], labels: List[str], k: int, label_to_id: dict, max_chunks_per_genome=100):
    print(f"\n--- Training with {k}-mers ---")
    os.makedirs(f"./results_k{k}", exist_ok=True)

    # This is where we are splitting the dataset into stratified sections.
    #   This methodology allows for the training, validation, and test splits to have a representative set of the data, given an overall
    #   non-uniform Distribution.
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in strat_split.split(sequences, labels):
        train_raw = [(sequences[i], labels[i]) for i in train_idx]
        val_raw = [(sequences[i], labels[i]) for i in val_idx]

    # This chunking of the data allows for easier processing of the millions of base pairs per genome sequence
    def chunk_and_label(data):
        all_chunks, all_labels = [], []
        for seq, label in data:
            chunks = sliding_window_kmers_limited(seq, k, MAX_LEN, STRIDE, max_chunks=max_chunks_per_genome)
            all_chunks.extend(chunks)
            all_labels.extend([label_to_id[label]] * len(chunks))
        return all_chunks, all_labels
    
    # adding in ability for Training to compute eval accuracy, instead of just eval loss
    #   This is important because Loss is not a very interpretable representation of the model performance. Accuracy can be taken at face value by comparison.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        return {
            "eval_accuracy": acc
        }

    # assigning the split data and the data tokenizer for the transformer's input
    train_chunks, train_chunk_labels = chunk_and_label(train_raw)
    val_chunks, val_chunk_labels = chunk_and_label(val_raw)

    tokenizer, model = load_base_bert(num_labels=len(label_to_id))
    model = truncate_transformer_weights(model, ratio=0.1)

    train_dataset = KmerWindowDataset(train_chunks, train_chunk_labels, tokenizer)
    val_dataset = KmerWindowDataset(val_chunks, val_chunk_labels, tokenizer)

    # Setting various training arguments here.
    training_args = TrainingArguments(
        output_dir=f"./results_k{k}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"./logs_k{k}",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )

    # Trainer is an absolutely wonderful utility from HuggingFace that allows for the typical training loop of ML to be handled
    #   much, MUCH more easily. Assuming you set everything up correctly, it is EZ-PZ from there.
    #   It also automatically detects and uses a systems GPU, assuming the torch library can see it, and your batch size fits into the GPU-memory.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # reporting highest accuracy and plotting validation accuracies across the epochs.
    acc = eval_results.get('eval_accuracy', 0.0)
    summary_path = "results_summary.txt"
    with open(summary_path, "a") as f:
        f.write(f"{k}-mer | Accuracy: {acc:.4f}\n")
    print(f"Logged accuracy to {summary_path}")

    plot_accuracy_curve(trainer, k)


# Main loop controlling the k-mer variable
def main(sequence_data_input: List[Tuple[str, str]]):
    # this code unzips the tuples into lists, as most machine learning code is optimized to work on lists.
    sequences, phyla = zip(*sequence_data_input)
    sequences, phyla = list(sequences), list(phyla)

    # converting the string phyla labels into unique integer IDs, which is the format the model can understand.
    # Also providing a reverse mapping of these IDs so that they can be converted back into the string labels during evaluation.
    label_to_id = {label: i for i, label in enumerate(sorted(set(phyla)))}
    id_to_label = {i: label for label, i in label_to_id.items()}

    # This loop handles the testing of the different values for the kmers
    # Also handles passing the IDs and the chunks of the genome (see above for chunking methodology)
    for k in KMER_SIZES:
        train_model(sequences, phyla, k, label_to_id, max_chunks_per_genome=MAX_CHUNKS_PER_GENOME)


# Main entry point for the program. Calls the main execution block of the training script.
if __name__ == "__main__":
    # Accepting the path to the data fodler as an argument
    data_path = sys.argv[1]
    # Assigning the ingested genome sequences to a local variable for pipeline preprocessing.
    sequence_data = genome_tuples(data_path)

    main(sequence_data)
