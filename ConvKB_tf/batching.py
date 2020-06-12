import sys, os
import logging
import numpy as np
import colorsys


# Current path
cur_path = os.path.dirname(os.path.realpath(os.path.basename(__file__)))

# Logging
logger = logging.getLogger("SRL Bench")
logger.setLevel(logging.DEBUG)
logger.propagate = False

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s\t(%(name)s)\t[%(levelname)s]\t%(message)s')

ch.setFormatter(formatter)

logger.addHandler(ch)


# Normal random tensor generation
def randn(*args): return np.random.randn(*args).astype('f')

class Batch_Loader(object):
    def __init__(self, train_triples, words_indexes, indexes_words, headTailSelector, \
                 entity2id, id2entity, relation2id, id2relation, batch_size=100, neg_ratio=1.0):
        self.train_triples = train_triples
        self.indexes = np.array(list(self.train_triples.keys())).astype(np.int32)
        self.values = np.array(list(self.train_triples.values())).astype(np.float32)
        self.batch_size = batch_size
        self.words_indexes = words_indexes
        self.indexes_words = indexes_words  # heads, relations, tails are also considered as words
        self.n_words = len(self.indexes_words)
        self.neg_ratio = int(neg_ratio)
        self.headTailSelector = headTailSelector
        self.relation2id = relation2id
        self.id2relation = id2relation
        self.entity2id = entity2id
        self.id2entity = id2entity

        self.indexes_rels = {}
        self.indexes_ents = {}
        for _word in self.words_indexes:
            index = self.words_indexes[_word]
            if _word in self.relation2id:
                self.indexes_rels[index] = _word
            elif _word in self.entity2id:
                self.indexes_ents[index] = _word

        self.new_triples_indexes = np.empty((self.batch_size * (self.neg_ratio + 1), 3)).astype(np.int32)
        self.new_triples_values = np.empty((self.batch_size * (self.neg_ratio + 1), 1)).astype(np.float32)

    def __call__(self):

        idxs = np.random.randint(0, len(self.values), self.batch_size)
        self.new_triples_indexes[:self.batch_size, :] = self.indexes[idxs, :]
        self.new_triples_values[:self.batch_size] = self.values[idxs, :]

        last_idx = self.batch_size

        if self.neg_ratio > 0:

            # Pre-sample everything, faster
            rdm_words = np.random.randint(0, self.n_words, last_idx * self.neg_ratio)
            # Pre copying everyting
            self.new_triples_indexes[last_idx:(last_idx * (self.neg_ratio + 1)), :] = np.tile(
                self.new_triples_indexes[:last_idx, :], (self.neg_ratio, 1))
            self.new_triples_values[last_idx:(last_idx * (self.neg_ratio + 1))] = np.tile(
                self.new_triples_values[:last_idx], (self.neg_ratio, 1))

            for i in range(last_idx):
                for j in range(self.neg_ratio):
                    cur_idx = i * self.neg_ratio + j
                    tmpRel = self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]]
                    tmpIndexRel = self.relation2id[tmpRel]
                    pr = self.headTailSelector[tmpIndexRel]

                    # Sample a random subject or object
                    if (np.random.randint(np.iinfo(np.int32).max) % 1000) > pr:
                        while (rdm_words[cur_idx] in self.indexes_rels or (
                                rdm_words[cur_idx], self.new_triples_indexes[last_idx + cur_idx, 1],
                                self.new_triples_indexes[last_idx + cur_idx, 2]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                        self.new_triples_indexes[last_idx + cur_idx, 0] = rdm_words[cur_idx]
                    else:
                        while (rdm_words[cur_idx] in self.indexes_rels or (
                                self.new_triples_indexes[last_idx + cur_idx, 0],
                                self.new_triples_indexes[last_idx + cur_idx, 1],
                                rdm_words[cur_idx]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                        self.new_triples_indexes[last_idx + cur_idx, 2] = rdm_words[cur_idx]

                    self.new_triples_values[last_idx + cur_idx] = [-1]

            last_idx += cur_idx + 1

        return self.new_triples_indexes[:last_idx, :], self.new_triples_values[:last_idx]

