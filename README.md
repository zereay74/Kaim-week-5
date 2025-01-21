# Project: Telegram Data Pipeline

This project provides a structured workflow for extracting, transforming, labeling, and fine-tuning NER models for text data from Telegram channels. The implementation is divided into five key tasks:

## Task 1: Telegram Data Ingestion
**Script/Notebook:** `Task_1_Telegram_Data_Ingestion.ipynb`

- **Objective:** Extract messages and media from specified Telegram channels.
- **Key Features:**
  - Handles multiple Telegram channels.
  - Saves extracted messages and associated media metadata.
  - Supports UTF-8-SIG encoding for multilingual text, including Amharic characters and emojis.
- **Output:** A CSV file (`telegram_data.csv`) containing extracted messages.

## Task 2: Data Labeling
**Script/Notebook:** `Task_2_Label_Data.ipynb`

- **Objective:** Label text data using the CoNLL format for NLP tasks.
- **Key Features:**
  - Reads the CSV file generated from Task 1.
  - Labels tokens based on predefined categories (e.g., product names, locations, prices).
  - Removes duplicates and ensures comprehensive labeling.
  - Supports UTF-8-SIG encoding to handle diverse characters.
- **Output:**
  - A labeled text file in CoNLL format (`labeled_data.txt`).
  - A labeled CSV file (`labeled_data.csv`).

## Task 3: Fine-Tuning NER Model
**Script/Notebook:** `Task_3_Fine_Tune_NER.ipynb`

- **Objective:** Fine-tune a Named Entity Recognition (NER) model to extract key entities (e.g., products, prices, locations) from Amharic Telegram messages.
- **Key Features:**
  - Fine-tunes pre-trained models like XLM-Roberta, DistilBERT, and mBERT.
  - Utilizes Hugging Face’s Trainer API for efficient model training.
  - Supports tokenized datasets aligned with CoNLL format.
- **Output:**
  - A fine-tuned NER model saved for future use.

## Task 4: Model Comparison & Selection
**Script/Notebook:** `Task_4_Model_Comparison.ipynb`

- **Objective:** Compare fine-tuned models and select the best-performing one for the entity extraction task.
- **Key Features:**
  - Evaluates models using metrics such as precision, recall, F1-Score, and loss.
  - Compares models like XLM-Roberta, DistilBERT, and mBERT for robustness and speed.
  - Identifies the best model for production based on comprehensive evaluation.
- **Output:**
  - A summary of model performance metrics.
  - Selection of the best-performing model for deployment.

## Task 5: Model Interpretability
**Script/Notebook:** `Task_5_Model_Interpretability.ipynb`

- **Objective:** Ensure transparency and trust in the NER system by explaining model predictions.
- **Key Features:**
  - Uses SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to interpret predictions.
  - Analyzes difficult cases, such as ambiguous text and overlapping entities.
  - Generates reports to summarize model decisions and identify areas for improvement.
- **Output:**
  - Interpretability visualizations and reports.

## How to Use
1. **Data Extraction:**
   - Configure your `.env` file with your Telegram API credentials.
   - Run `Task_1_Telegram_Data_Ingestion.ipynb` to extract Telegram data.
2. **Data Labeling:**
   - Ensure `telegram_data.csv` is present in your working directory.
   - Run `Task_2_Label_Data.ipynb` to label the data.
3. **Model Fine-Tuning:**
   - Use `Task_3_Fine_Tune_NER.ipynb` to fine-tune pre-trained NER models.
4. **Model Comparison:**
   - Run `Task_4_Model_Comparison.ipynb` to evaluate and compare fine-tuned models.
5. **Model Interpretability:**
   - Use `Task_5_Model_Interpretability.ipynb` to analyze model predictions and ensure transparency.

## Requirements
- Python 3.8+
- Required libraries: `pandas`, `telethon`, `transformers`, `datasets`, `seqeval`, `shap`, `lime`, `python-dotenv`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Outputs
- `telegram_data.csv`: Raw Telegram data.
- `labeled_data.txt`: Labeled data in CoNLL format.
- `labeled_data.csv`: Labeled data in CSV format.
- `fine_tuned_model/`: Directory containing the fine-tuned NER model.
- `model_performance_summary.csv`: A summary of model evaluation metrics.
- Interpretability visualizations and reports.

