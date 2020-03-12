import pandas as pd
import numpy as np
import typing as t

from nlp_tools import utils
from nlp_tools.dump import dump_word_index, dump_merged_word_matrix


class IndexedRepresentation:
    index_columns: t.List[str] = ['lem', 'index']
    index: pd.DataFrame
    index_len: int
    data: t.Dict[str, np.array]
    data_merged: t.Dict[str, np.array]

    def __init__(self, data):
        self.index = self.build_index(data)
        self.index_len = self.index.shape[0]

    def build_index(self, data) -> pd.DataFrame:
        index = dict()
        idx = 0
        for intent in data:
            for example in data[intent]:
                for lem in example:
                    if index.get(lem) is None:
                        index[lem] = idx
                        idx += 1
        index_dict = [{self.index_columns[0]: k, self.index_columns[1]: index[k]} for k in index]
        dump_word_index(index_dict)

        return pd.DataFrame(index_dict, columns=self.index_columns)

    def get_index_from_lem(self, lem) -> int:
        idx = self.index.loc[self.index[self.index_columns[0]] == lem, self.index_columns[1]]
        if idx.empty:
            return None
        else:
            return idx.values[0]

    def get_matrix_from_index(self, idx) -> np.array:
        matrix = np.zeros(self.index_len)
        matrix[idx] = 1
        return matrix

    def get_matrix_from_lem(self, lem) -> np.array:
        idx = self.get_index_from_lem(lem)
        if idx is None:
            return idx
        else:
            return self.get_matrix_from_index(idx)

    def process_words_matrix(self, example) -> np.array:
        return np.array(
            [self.get_matrix_from_lem(l) for l in example if not self.get_matrix_from_lem(l) is None])


class MatrixRepresentation(IndexedRepresentation):
    data: t.Dict[str, np.array]

    def __init__(self, data):
        IndexedRepresentation.__init__(self, data)
        self.data = self.process_words_matrix_for_all(data)

    def process_words_matrix_for_all(self, data) -> t.Dict[str, np.array]:
        get_matrix = lambda examples: np.array(
            [self.process_words_matrix(ex) for ex in examples])
        return utils.apply_for_each_key(data, get_matrix)

    def process_new_data(self, lems) -> np.array:
        return self.process_words_matrix(lems)


class MergedMatrixRepresentation(IndexedRepresentation):
    data: t.Dict[str, np.array]

    def __init__(self, data):
        IndexedRepresentation.__init__(self, data)
        self.data = self.process_merged_sentences_for_all(data)
        self.save_merged_matrix(self.data)

    def save_merged_matrix(self, matrix):
        as_list = lambda np_array: np_array.tolist()
        serializable_matrix = utils.apply_for_each_key(matrix, as_list)
        dump_merged_word_matrix(serializable_matrix)

    def process_words_merged_matrix(self, example) -> np.array:
        matrix = self.process_words_matrix(example)
        return sum(list(matrix))

    def process_merged_sentences_for_all(self, data) -> t.Dict[str, np.array]:
        get_matrix = lambda examples: np.array(
            [self.process_words_merged_matrix(ex) for ex in examples])
        return utils.apply_for_each_key(data, get_matrix)

    def process_new_data(self, lems) -> np.array:
        return self.process_words_merged_matrix(lems)
