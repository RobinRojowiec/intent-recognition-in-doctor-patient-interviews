"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: search_engine
Date: 26.04.2019

"""
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from whoosh import scoring
from whoosh.analysis import Token, Tokenizer
from whoosh.fields import *
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from whoosh.query import Every


class BertAnalyzer(Tokenizer):
    def __init__(self):
        # Alternative: BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=self.cased)
        self.cased = True
        self.remove_stopwords = True
        self.stop_words = set(stopwords.words('german'))

    def __call__(self, *args, **kwargs):
        sentence = args[0]
        if self.remove_stopwords:
            sentence = self.filter_stop_words(sentence)
        token = Token(removestops=False, mode='', positions=False, chars=False)
        for word_piece in word_tokenize(sentence.lower(), language='german'):
            if self.remove_stopwords:
                if word_piece not in self.stop_words:
                    token.text = word_piece
                    yield token
            else:
                token.text = word_piece
                yield token

    def __eq__(self, other):
        return False

    def filter_stop_words(self, sent):
        """
        Returns a filtered sentence without stopwords
        :param sent:
        :return:
        """
        word_tokens = word_tokenize(sent)
        filtered_sentence = [w for w in word_tokens if not w.lower() in self.stop_words]
        return ' '.join(filtered_sentence).strip()


class SearchEngine:
    def __init__(self, index_path: str, read_only=True):
        self.index_path = index_path
        self.analyzer = BertAnalyzer()

        if not os.path.exists(index_path):
            os.mkdir(index_path)

        if not read_only:
            schema = Schema(utterance=TEXT(analyzer=self.analyzer, stored=True, phrase=False),
                            previous_classes=KEYWORD(stored=True),
                            classes=KEYWORD(stored=True),
                            position=NUMERIC(stored=True))
            self.index = create_in(self.index_path, schema)
        else:
            self.index = open_dir(self.index_path)

    def add_documents(self, docs: list):
        """
        Indexes a list of json documents
        :param docs:
        :return:
        """
        with self.index.writer() as writer:
            for doc in docs:
                for row in doc:
                    writer.add_document(**row)

    def get_doc_count(self):
        with self.index.reader() as reader:
            return reader.doc_count()

    def search(self, query: str, limit=10, scoring_function=scoring.BM25F()):
        """
        Search documents
        :param scoring_function:
        :param query:
        :param limit:
        :return:
        """
        result_list = []
        tokens = list(
            set(filter(lambda token: token not in ['and', 'or'], [token.text for token in self.analyzer(query)])))
        query = ' OR '.join(tokens)
        with self.index.searcher(weighting=scoring_function) as searcher:
            query = MultifieldParser(["utterance"], self.index.schema).parse(query)
            scored_docs = searcher.search(query, limit=limit)
            for scored in scored_docs:
                result = dict()
                for key in scored.iterkeys():
                    result[key] = scored[key]
                result['score'] = scored.score
                result_list.append(result)
            return result_list
        return None

    def clear(self):
        with self.index.writer() as writer:
            writer.delete_by_query(Every())
