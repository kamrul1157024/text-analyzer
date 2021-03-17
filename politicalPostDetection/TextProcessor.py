from bengali_stemmer.rafikamal2014 import RafiStemmer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from .apps import PoliticalPostDetectionConfig


class BanglaTextPreprocessor:

    def __init__(self):
        self.stemmer = RafiStemmer()
        self.banglaStopWordsMap = PoliticalPostDetectionConfig.banglaStopWordsMap

    def process_text(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        english_pattern = re.compile('[a-zA-Z0-9]+', flags=re.I)
        begali_digits = re.compile(u'[\u09E6-\u09EF]+', flags=re.UNICODE)
        punctuation = re.compile(u'[.,!?\\-\u0964]', flags=re.UNICODE)
        text = punctuation.sub(r'', text)
        text = begali_digits.sub(r'', text)
        text = emoji_pattern.sub(r'', text)
        text = english_pattern.sub(r'', text)
        text = re.sub(r'\s+', ' ', text)
        word_tokens = word_tokenize(text)
        filtered_sequence_without_stopwords = [w for w in word_tokens if not self.banglaStopWordsMap.get(w, False)]
        filtered_sequence = [self.stemmer.stem_word(w) for w in filtered_sequence_without_stopwords]
        text = ' '.join(filtered_sequence)
        return text


class EnglishTextPreprocessor:
    def __init__(self):
        pass

    def process_text(self,text):
        text = text.lower().replace('\n', ' ').replace('\r', '').strip()
        text = re.sub(' +', ' ', text)
        text = re.sub('[^a-z\s]*', '', text)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sequence = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(filtered_sequence)
        return text
