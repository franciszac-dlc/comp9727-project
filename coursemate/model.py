from collections import defaultdict
from itertools import combinations
from abc import ABC, abstractmethod
from typing import Callable, Any

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from coursemate.dataset import Dataset


class RecommenderModel(ABC):
    @abstractmethod
    def fit(self, training_data: Any):
        return NotImplemented

    @abstractmethod
    def recommend(self, prev_courses: Any, k: int = 5):
        return NotImplemented


class AssociationMiningModel(RecommenderModel):
    def __init__(self, cutoff: int, course_set: pd.DataFrame):
        self.cutoff = cutoff
        self.course_to_index = {
            c: ndx
            for ndx, c in enumerate(course_set.index)
        }
        self.index_to_course = {
            ndx: c
            for ndx, c in enumerate(course_set.index)
        }

    @staticmethod
    def generate_subsequences(sequence_dict, k):
        subseq_counts = defaultdict(int)
        for _, seq in tqdm(sequence_dict.items()):
            for comb in combinations(seq, k):
                subseq_counts[comb] += 1

        return subseq_counts

    @staticmethod
    def generate_antecedents(prev_courses: tuple):
        return set(combinations(prev_courses, 1)) | set(combinations(prev_courses, 2))

    def get_sequences_data(self, train_X, train_y):
        full_seqs = {}
        for (name, prevs_c), next_c in zip(train_X, train_y):
            full_seqs[name] = (*prevs_c, next_c)

        full_seqs_ndx = {}
        for n, cs in full_seqs.items():
            full_seqs_ndx[n] = tuple(self.course_to_index[c] for c in cs)

        seq1 = AssociationMiningModel.generate_subsequences(full_seqs_ndx, 1)
        seq2 = AssociationMiningModel.generate_subsequences(full_seqs_ndx, 2)
        seq3 = AssociationMiningModel.generate_subsequences(full_seqs_ndx, 3)

        print(f"1-subsequences: {len(seq1)}")
        print(f"2-subsequences: {len(seq2)}")
        print(f"3-subsequences: {len(seq3)}")

        df_seq1 = pd.DataFrame(seq1.items(), columns=['subsequence', 'count'])
        df_seq2 = pd.DataFrame(seq2.items(), columns=['subsequence', 'count'])
        df_seq3 = pd.DataFrame(seq3.items(), columns=['subsequence', 'count'])

        df_seq = pd.concat([df_seq1, df_seq2, df_seq3])

        self.df_seq_dict = {**seq1, **seq2, **seq3}
        self.user_count = len(full_seqs_ndx)

        print(df_seq.shape)
        return df_seq

    def fit(self, training_data: pd.DataFrame):
        _df = training_data[(training_data['count'] > self.cutoff) & (training_data['subsequence'].apply(len) > 1)].copy()

        _df['antecedent'] = _df['subsequence'].apply(lambda x: x[:-1])
        _df['consequent'] = _df['subsequence'].apply(lambda x: (x[-1],))
        _df['antecedent_count'] = _df['antecedent'].apply(lambda x: self.df_seq_dict[x] if x in self.df_seq_dict else 0)
        _df['consequent_count'] = _df['consequent'].apply(lambda x: self.df_seq_dict[x] if x in self.df_seq_dict else 0)
        
        _df['confidence'] = _df['count'] / _df['antecedent_count']
        _df['lift'] = _df['confidence'] / (_df['antecedent_count'] / self.user_count)

        self.frequent_subseqs = _df

    def recommend(self, prev_courses: tuple, k: int = 5):
        prev_courses_ndx = tuple(self.course_to_index[c] for c in prev_courses)
        antecedents = AssociationMiningModel.generate_antecedents(prev_courses_ndx)

        _candidate_set = self.frequent_subseqs[self.frequent_subseqs['antecedent'].isin(antecedents)][['consequent', 'confidence']] \
                            .sort_values(by='confidence', ascending=False) \
                            .drop_duplicates(subset=['consequent'])
        _results = _candidate_set.head(k)['consequent'].apply(lambda x: x[0]).values

        return tuple(self.index_to_course[i] for i in _results)
