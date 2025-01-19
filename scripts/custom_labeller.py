import spacy
import pandas as pd
from spacy import Language

def train_custom_ner(data, output_model_dir):
    """
    Train a custom NER model using spaCy.

    :param data: A list of training examples in the format [(text, annotations)].
                 Example: [("Text", {"entities": [(start, end, label)]})]
    :param output_model_dir: Directory to save the trained model.
    """
    from spacy.training.example import Example
    from spacy.util import minibatch
    from pathlib import Path

    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()
    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in data]

    for epoch in range(200):
        losses = {}
        batches = minibatch(examples, size=8)
        for batch in batches:
            nlp.update(batch, drop=0.5, losses=losses)
        print(f"Losses at epoch {epoch}: {losses}")

    Path(output_model_dir).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_model_dir)
    print(f"Model saved to {output_model_dir}")

def label_data_with_ner(ner_model: Language, dataframe: pd.DataFrame, text_column: str, output_csv_path: str):
    """
    Labels the input dataset using an NER model and saves the output to a CSV file.

    :param ner_model: The loaded spaCy NER model.
    :param dataframe: The input dataframe containing text data.
    :param text_column: The column containing text to process.
    :param output_csv_path: Path to save the labeled dataset in CSV format.
    """
    # Ensure all values in the text column are strings
    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)

    labeled_data = []

    for _, row in dataframe.iterrows():
        text = row[text_column]
        doc = ner_model(text)

        entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        labeled_data.append({"text": text, "entities": entities})

    # Save the labeled data to CSV
    labeled_df = pd.DataFrame(labeled_data)
    labeled_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
   

'''
if __name__ == "__main__":
    # Example usage

    # Load your scraped data
    input_data = pd.read_csv("telegram_data.csv", encoding="utf-8-sig")
    text_column = "Message"

    # Prepare training data (manually annotated for demonstration purposes)
    training_data = [
        ("LIFESTAR receiver is available for 5000 birr.", {"entities": [(0, 8, "B-Product"), (9, 17, "I-Product"), (36, 40, "I-PRICE")]}),
        ("Addis Ababa is a great city.", {"entities": [(0, 11, "B-LOC")]})
    ]

    # Train and save the model
    model_output_dir = "trained_ner_model"
    train_custom_ner(training_data, model_output_dir)

    # Load the trained model
    ner_model = load_trained_model(model_output_dir)

    # Label the dataset
    output_csv_path = "labeled_ner_data.csv"
    label_data_with_ner(ner_model, input_data, text_column, output_csv_path)
''' 