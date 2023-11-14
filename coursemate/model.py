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


class ItemBasedCF(RecommenderModel):
    def __init__(self, course_set: pd.DataFrame):
        self.course_similarity_matrix = None
        self.course_set = course_set

    def fit(self, training_data: pd.DataFrame):
        # Prepare the data in a user-item matrix format
        user_item_matrix = self.create_user_item_matrix(training_data)
        # Calculate the similarity matrix
        self.course_similarity_matrix = self.calculate_similarity(user_item_matrix)

    def create_user_item_matrix(self, training_data):
        # Pivot table to create a matrix of users and courses with ratings as values
        user_item_matrix = training_data.pivot_table(index='reviewers', columns='course_id', values='rating')
        user_item_matrix = user_item_matrix.fillna(0)
        return user_item_matrix

    def calculate_similarity(self, user_item_matrix):
        # Use cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(user_item_matrix.T)
        np.fill_diagonal(similarity_matrix, 0)
        return pd.DataFrame(similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    def recommend(self, prev_courses: tuple, k: int = 5):
        recommendations = self.generate_recommendations(prev_courses, k)
        return recommendations

    def generate_recommendations(self, prev_courses, k):
        # Get the similarity scores for the previous courses
        course_similarities = self.course_similarity_matrix.loc[prev_courses].sum().sort_values(ascending=False)
        # Exclude already rated courses
        course_similarities = course_similarities[~course_similarities.index.isin(prev_courses)]
        # Get top-k courses
        top_k_courses = course_similarities.head(k).index.tolist()
        return top_k_courses


class UserBasedCF(RecommenderModel):
    def __init__(self):
        self.user_similarity_matrix = None
        self.train_ratings = None  # Store training data here

    def fit(self, training_data: pd.DataFrame):
        # Store the training data
        self.train_ratings = training_data
        # Create a user-item matrix
        user_item_matrix = self.create_user_item_matrix(training_data)
        # Calculate the similarity matrix
        self.user_similarity_matrix = self.calculate_similarity(user_item_matrix)

    def create_user_item_matrix(self, training_data):
        # Pivot table to create a matrix of users and courses with ratings as values
        user_item_matrix = training_data.pivot_table(index='reviewers', columns='course_id', values='rating')
        user_item_matrix = user_item_matrix.fillna(0)
        return user_item_matrix

    def calculate_similarity(self, user_item_matrix):
        # Use cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(user_item_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        return pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    def recommend(self, user_id, k: int = 5):
        recommendations = self.generate_recommendations(user_id, k)
        return recommendations

    def generate_recommendations(self, user_id, k):
        # Get the top similar users
        similar_users = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False).head(k).index

        # Filter the training data to only include ratings from similar users
        similar_users_ratings = self.train_ratings[self.train_ratings['reviewers'].isin(similar_users)]

        # Aggregate these ratings to get an average rating for each course
        aggregated_ratings = similar_users_ratings.groupby('course_id')['rating'].mean().sort_values(ascending=False)

        # Get top-k recommendations (courses with the highest average rating)
        top_k_courses = aggregated_ratings.head(k).index.tolist()
        return top_k_courses

