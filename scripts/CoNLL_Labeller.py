import pandas as pd
import os
import re

class CoNLLLabeler:
    def __init__(self, dataframe, text_column, output_txt, output_csv):
        """
        Initializes the CoNLLLabeler with a dataframe, the column to process, and output file paths.

        :param dataframe: The input dataframe containing text data.
        :param text_column: The name of the column containing text to label.
        :param output_txt: The path to save the labeled CoNLL format text file.
        :param output_csv: The path to save the labeled data in CSV format.
        """
        self.dataframe = dataframe
        self.text_column = text_column
        self.output_txt = output_txt
        self.output_csv = output_csv
        self.entities = {
            "B-Product": list(set([
                'LIFESTAR', 'ላይፍስታር', 'XCRUISER', 'ኢክስክሩሰር', 'HDMI', 'ኤችዲ ኤም አይ', 'GOLDSTAR', 'ጎልድስታር', 'SUPERSTAR', 'ሱፐርስታር',
                'DEGOL', 'Washing', 'Sattelite', 'ሳተላይት', 'Websprix Internet', 'ETV PRO', '57E SPTV', 'Forever', 'MyHD',
                'EthioShare', 'Super G-Share', 'SDS', 'VIP', 'TV STICK', 'Mango', 'MI 2K', 'Q5 PLUS', 'SATLOCKER'
            ])),
            "I-Product": list(set([
                'Receiver', 'ሪሲቨር', 'TV', 'ቲቪ', 'ANDROID TV', 'አንድሮይድ ቲቪ', 'SPLITTER', 'ስፕሊተር', 'Dish', 'ድሽ', 'Server',
                'LNB', 'ኤልኤንቢ', 'ሰርቨር', 'Machine', 'HDMI', 'ANDROID BOX', 'Finder', 'ፋይደር', 'አዳፕተሮች', 'አዳፕተር', 'ADAPTER',
                'FINDER', 'ፍይንደር', '2500W', 'Coin', 'CAMERA', '5600D', '5500D', 'ስክሪን', 'ባትሪ', 'ቻርጀር', 'RF TUNER',
                'ቻርጀር አይሲ', 'ኤል ኤን ቢ አይሲ', 'አይሲዎች', 'IC', 'Advanced Programmer', 'Eeprom Software መጫኛ',
                '7600HD', '7700HD', '7800HD', '7900HD', '9200HD', 'SMART', '9300HD', '1000HD', '2000HD', '3000HD', '4000HD',
                'GOLD', 'mini', 'V8', 'Super', '95HD', '96HD', '97HD', '98HD', '9595HD', '4K', '9090HD', 'Diamond',
                '6060HD', '8080HD', '8585HD++', '9999HD', '9000', 'ghost', '7200HD', '7500HD', '8600HD', '8800HD', '6565HD', 'Mega', '6464HD'
            ])),
            "B-LOC": list(set([
                'Addis Ababa', 'አዲስ አበባ', 'Merkato', 'መርካቶ', 'Raguel', 'ራጉኤል', 'Anwar Meskid', 'አንዋር መስኪድ', 'Bole', 'ቦሌ',
                'Megeneagna', 'መገናኛ', 'Piyasa', 'ፒያሳ', 'ሀረር', 'አዳማ', 'ጎንደር', 'ደብረብርን', 'ባህር ዳር', 'ደሴ'
            ])),
            "I-LOC": list(set([
                'Abeba', 'አበባ', 'መስኪድ'
            ])),
            "B-PRICE": list(set([
                'ዋጋ', 'ብር', 'Birr', '$'
            ])),
            "I-PRICE": []  # Populated dynamically
        }

    def generate_price_entities(self):
        """
        Dynamically generate I-PRICE entities for numeric price patterns followed by valid currency symbols.
        """
        patterns = [
            r'\b\d+ብር\b', r'\b\d+ETB\b', r'\b\d+\$\b', r'\b\d+Birr\b'
        ]
        self.entities["I-PRICE"].extend(patterns)

    def label_text(self, text):
        """
        Labels the text using CoNLL format based on predefined entities.

        :param text: The input text to label.
        :return: A list of tuples, each containing a token and its label.
        """
        tokens = text.split()
        labels = ["O"] * len(tokens)

        for entity_type, entity_list in self.entities.items():
            if entity_type != "I-PRICE":
                for entity in entity_list:
                    entity_tokens = entity.split()
                    for i in range(len(tokens)):
                        if [t.lower() for t in tokens[i:i + len(entity_tokens)]] == [e.lower() for e in entity_tokens]:
                            if all(label == "O" for label in labels[i:i + len(entity_tokens)]):
                                labels[i] = f"B-{entity_type.split('-')[1]}"
                                for j in range(1, len(entity_tokens)):
                                    labels[i + j] = f"I-{entity_type.split('-')[1]}"
            else:  # Handle I-PRICE patterns
                for pattern in entity_list:
                    for i, token in enumerate(tokens):
                        if re.match(pattern, token):
                            if labels[i] == "O":
                                labels[i] = "I-PRICE"

        # Remove duplicate labels
        seen_labels = {}
        for i, token in enumerate(tokens):
            if token.lower() in seen_labels:
                labels[i] = "O"
            else:
                seen_labels[token.lower()] = labels[i]

        return list(zip(tokens, labels))

    def save_to_conll_and_csv(self):
        """
        Processes the dataframe and saves labeled data in both CoNLL and CSV formats.
        """
        labeled_data = []

        for _, row in self.dataframe.iterrows():
            text = str(row[self.text_column])
            labeled_tokens = self.label_text(text)
            labeled_data.extend(labeled_tokens)
            labeled_data.append(("", ""))  # Add a blank line for sentence separation

        # Save to text file (CoNLL format)
        with open(self.output_txt, 'w', encoding='utf-8-sig') as txt_file:
            for token, label in labeled_data:
                if token and label:
                    txt_file.write(f"{token} {label}\n")
                else:
                    txt_file.write("\n")

        # Save to CSV file
        labeled_df = pd.DataFrame(
            [item for item in labeled_data if item != ("", "")],
            columns=["Token", "Label"]
        )
        labeled_df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')



   

'''
if __name__ == '__main__':
    # Example usage

    # Load your scraped data into a DataFrame (assuming CSV format for simplicity)
    data = pd.read_csv('telegram_data.csv', encoding='utf-8-sig')

    # Specify the column to label
    text_column = 'Message'

    # Output files for CoNLL format and CSV format
    output_txt = 'labeled_data.txt'
    output_csv = 'labeled_data.csv'

    # Initialize and run the labeler
    labeler = CoNLLLabeler(dataframe=data, text_column=text_column, output_txt=output_txt, output_csv=output_csv)
    labeler.save_to_conll_and_csv()

    print(f"Labeled data has been saved to {output_txt} and {output_csv}.")
'''