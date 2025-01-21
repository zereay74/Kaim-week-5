import os
import numpy as np
from datasets import load_metric, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# Step 1: Load the Pre-trained Model and Tokenizer
def load_model_and_tokenizer(model_name: str):
    """
    Load a pre-trained model and tokenizer for NER tasks.
    :param model_name: Name of the pre-trained model (e.g., "xlm-roberta-base").
    :return: tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=6)  # Adjust num_labels
    return tokenizer, model

# Step 2: Preprocess Tokenized Dataset
def preprocess_dataset(tokenizer, data_path: str, max_length: int = 128):
    """
    Preprocess the dataset into tokenized format suitable for NER tasks.
    :param tokenizer: Tokenizer object
    :param data_path: Path to the labeled dataset
    :param max_length: Maximum token sequence length
    :return: Processed Dataset object
    """
    # Load the dataset from the file
    def read_conll(file_path):
        sentences, labels = [], []
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            words, tags = [], []
            for line in file:
                if line.strip() == "":
                    if words:
                        sentences.append(words)
                        labels.append(tags)
                        words, tags = [], []
                else:
                    word, tag = line.strip().split()
                    words.append(word)
                    tags.append(tag)
            if words:  # Append last sentence if not empty
                sentences.append(words)
                labels.append(tags)
        return sentences, labels

    sentences, labels = read_conll(data_path)

    # Encode the sentences and align labels
    def tokenize_and_align_labels(sentences, labels):
        tokenized_inputs = tokenizer(
            sentences,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        label_all_tokens = True
        aligned_labels = []

        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    aligned_label_ids.append(-100)
                elif label_all_tokens or word_id != word_ids[word_ids.index(word_id) - 1]:
                    aligned_label_ids.append(label[i][word_id])
                else:
                    aligned_label_ids.append(-100)
            aligned_labels.append(aligned_label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    # Convert to Hugging Face Dataset
    tokenized_dataset = Dataset.from_dict(
        tokenize_and_align_labels(sentences, labels)
    )
    return tokenized_dataset

# Step 3: Set Up Training Arguments
def setup_training_args(output_dir: str):
    """
    Set up training arguments.
    :param output_dir: Directory to save the fine-tuned model
    :return: TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
    )

# Step 4: Train the Model
def train_model(model, tokenizer, train_dataset, val_dataset, training_args):
    """
    Fine-tune the NER model.
    :param model: Pre-trained model
    :param tokenizer: Tokenizer
    :param train_dataset: Tokenized training dataset
    :param val_dataset: Tokenized validation dataset
    :param training_args: TrainingArguments object
    """
    metric = load_metric("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        # Remove ignored index (-100)
        true_predictions = [
            [p for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for l in label if l != -100] for label in labels
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

# Step 5: Save the Model
def save_model(trainer, save_path: str):
    """
    Save the fine-tuned model for future use.
    :param trainer: Trainer object
    :param save_path: Path to save the model
    """
    trainer.save_model(save_path)

'''
# Main Function
if __name__ == "__main__":
    # Define paths
    model_name = "xlm-roberta-base"  # Change to "bert-tiny-amharic" or "afroxmlr" if required
    data_path = "labeled_tokenized_dataset.txt"
    output_dir = "./fine_tuned_model"

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name)

    # Preprocess dataset
    dataset = preprocess_dataset(tokenizer, data_path)
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

    # Set up training arguments
    training_args = setup_training_args(output_dir)

    # Fine-tune the model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, training_args)

    # Save the model
    save_model(trainer, output_dir)
''' 