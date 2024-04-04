import re
import string

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from numpy import ndarray

nltk.download('stopwords')
nltk.download('wordnet')


class TextPipeline:
    def __init__(self, text):
        self.__wnl = WordNetLemmatizer()
        self.__stopwords_str = ", ".join(nltk.corpus.stopwords.words('english'))
        self.text = text

    def _preprocess(self):
        self.text = self.text.lower()
        self.text = re.sub("<.+?/?>", "", self.text)
        self.text = self.text.translate(str.maketrans("", "", string.punctuation))
        self.text = re.sub(r"\d+", "", self.text)
        self.text = re.sub(r"  ", " ", self.text)

    def _remove_stopwords(self):
        new_text = []
        for token in self.text.split():
            if not token in self.__stopwords_str:
                new_text.append(token)
        self.text = " ".join(new_text)

    def _lemmatize(self):
        new_text = []
        for word in self.text.split():
            word = self.__wnl.lemmatize(word)
            new_text.append(word)
        self.text = " ".join(new_text)

    def preprocess_text(self) -> str:
        self._preprocess()
        self._remove_stopwords()
        self._lemmatize()
        print(self.text)
        return self.text

    def vectorize_text(self) -> ndarray:
        print(self.text)
        with open("app/artifacts/vocab.txt", "r") as file:
            vocab_d = file.readlines()
        vocab = []
        for w in vocab_d:
            try:
                vocab.append(w.strip())
            except Exception as e:
                vocab.append(str(w).strip())
        vector_array = []
        vector = np.zeros(len(vocab))
        words = self.text.split()
        for i in range(len(vocab)):
            if vocab[i] in words:
                vector[i] = 1
        print(len(vector))
        vector_array.append(vector)
        np_array = np.asarray(vector_array, dtype=np.float32)
        return np_array

