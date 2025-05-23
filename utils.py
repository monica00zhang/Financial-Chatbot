import re
import pickle
import json

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def load_buffett_letters():
    with open("Dataset/company_sample.json", "r", encoding="utf-8") as f:
        company_part = json.load(f)

    with open("Dataset/meta_letter.json", "r", encoding="utf-8") as f:
        letters = json.load(f)

    with open("Dataset/indicator_exa.json", "r", encoding="utf-8") as f:
        financial_term = json.load(f)

    return [letters, company_part, financial_term]




def extract_info(text):
    extracted = {"signal": None, "date": None, "company": None}

    words = text.replace(".", "").split(", ")

    for phrase in words:
        parts = phrase.split()
        if "signal" in phrase:
            extracted["signal"] = " ".join(parts[1:parts.index("signal")])
        elif "date" in phrase:
            extracted["date"] = parts[-1]  # "2022"
        elif "company" in phrase:
            extracted["company"] = parts[-1]  # "microsoft"

    return extracted

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence