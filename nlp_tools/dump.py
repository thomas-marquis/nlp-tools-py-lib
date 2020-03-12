import json
import io

from nlp_tools.constants import *


def dump_json(data, file_name, output_path=OUT_PATH):
    with io.open(output_path + file_name, "w", encoding="utf8") as out:
        data = json.dumps(data, ensure_ascii=False)
        out.write(data)


def dump_stop_words(stop_words):
    dump_json(stop_words, STOP_WORDS_DUMP)


def dump_word_index(word_index):
    dump_json(word_index, INDEX_DUMP)


def dump_merged_word_matrix(word_matrix):
    dump_json(word_matrix, MERGED_W_MATRIX_DUMP)