# Project: Telegram Data Pipeline

This project provides a structured workflow for extracting, transforming, and labeling text data from Telegram channels. The implementation is divided into two key tasks:

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

## How to Use
1. **Data Extraction:**
   - Configure your `.env` file with your Telegram API credentials.
   - Run `Task_1_Telegram_Data_Ingestion.ipynb` to extract Telegram data.
2. **Data Labeling:**
   - Ensure `telegram_data.csv` is present in your working directory.
   - Run `Task_2_Label_Data.ipynb` to label the data.

## Requirements
- Python 3.8+
- Required libraries: `pandas`, `telethon`, `python-dotenv`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Outputs
- `telegram_data.csv`: Raw Telegram data.
- `labeled_data.txt`: Labeled data in CoNLL format.
- `labeled_data.csv`: Labeled data in CSV format.



