from collections import defaultdict
from itertools import combinations
from abc import ABC, abstractmethod
from typing import Callable, Any, Union

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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
        self.course_to_index = {c: ndx for ndx, c in enumerate(course_set.index)}
        self.index_to_course = {ndx: c for ndx, c in enumerate(course_set.index)}

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

        df_seq1 = pd.DataFrame(seq1.items(), columns=["subsequence", "count"])
        df_seq2 = pd.DataFrame(seq2.items(), columns=["subsequence", "count"])
        df_seq3 = pd.DataFrame(seq3.items(), columns=["subsequence", "count"])

        df_seq = pd.concat([df_seq1, df_seq2, df_seq3])

        self.df_seq_dict = {**seq1, **seq2, **seq3}
        self.user_count = len(full_seqs_ndx)

        print(df_seq.shape)
        return df_seq

    def fit(self, training_data: pd.DataFrame):
        _df = training_data[
            (training_data["count"] > self.cutoff)
            & (training_data["subsequence"].apply(len) > 1)
        ].copy()

        _df["antecedent"] = _df["subsequence"].apply(lambda x: x[:-1])
        _df["consequent"] = _df["subsequence"].apply(lambda x: (x[-1],))
        _df["antecedent_count"] = _df["antecedent"].apply(
            lambda x: self.df_seq_dict[x] if x in self.df_seq_dict else 0
        )
        _df["consequent_count"] = _df["consequent"].apply(
            lambda x: self.df_seq_dict[x] if x in self.df_seq_dict else 0
        )

        _df["confidence"] = _df["count"] / _df["antecedent_count"]
        _df["lift"] = _df["confidence"] / (_df["antecedent_count"] / self.user_count)

        self.frequent_subseqs = _df

    def recommend(self, prev_courses: tuple, k: int = 5):
        prev_courses_ndx = tuple(self.course_to_index[c] for c in prev_courses)
        antecedents = AssociationMiningModel.generate_antecedents(prev_courses_ndx)

        _candidate_set = (
            self.frequent_subseqs[
                self.frequent_subseqs["antecedent"].isin(antecedents)
            ][["consequent", "confidence"]]
            .sort_values(by="confidence", ascending=False)
            .drop_duplicates(subset=["consequent"])
        )
        _results = _candidate_set.head(k)["consequent"].apply(lambda x: x[0]).values

        return tuple(self.index_to_course[i] for i in _results)


class ContentBasedModel(RecommenderModel):
    def __init__(
        self,
        course_set: pd.DataFrame,
        Vectorizer: Union[TfidfVectorizer, CountVectorizer] = TfidfVectorizer,
        n_features: int = 10000,
    ):
        """
        Initializes the Content Based recommender model.

        Parameters:
        course_set (pd.DataFrame): The set of courses available for recommendation.
        Vectorizer: The vectorizer to be used for transforming course descriptions and skills into vectors.
        n_features (int): The maximum number of features to be used by the vectorizer.
        """
        self.course_set = course_set

        # Move to the fit function?
        self.vectorizer = Vectorizer(max_features=n_features)
        self.vectorizer.fit(self.course_set["description"] + self.course_set["skills"])

        self.course_vectors = {}
        for id, row in self.course_set.iterrows():
            self.course_vectors[id] = self.vectorizer.transform(
                [row["description"] + row["skills"]]
            )

    def fit(self, training_data: Any):
        pass

    def recommend(self, prev_courses: tuple, k: int = 5):
        """
        Recommends courses based on the courses previously taken by the user.

        Parameters:
        prev_courses (tuple): The courses previously taken by the user.
        k (int): The number of courses to recommend. Default is 5.

        Returns:
        list: The IDs of the recommended courses.
        """
        #indexes = ()

        #for i in range(len(prev_courses)):
        #    indexes = indexes + (self.course_set.loc[prev_courses[i]].name,)
        #print(indexes)
        user_reviews_skills = self.course_set[self.course_set.index.isin(prev_courses)]["skills"]
        user_reviews_description = self.course_set[self.course_set.index.isin(prev_courses)]["description"]
        user_reviews_combined = user_reviews_skills + " " + user_reviews_description

        user_vector = self.vectorizer.transform(user_reviews_combined)

        most_similar_courses = []
        for other_course_id in self.course_set.index:
            if other_course_id in prev_courses:
                continue

            course_vector = self.course_vectors[other_course_id]
            normalized_user_vector = normalize(user_vector)
            normalized_course_vector = normalize(course_vector)
            similarity = cosine_similarity(
                normalized_user_vector, normalized_course_vector
            )
            most_similar_courses.append((other_course_id, similarity.mean()))

        most_similar_courses.sort(key=lambda x: x[1], reverse=True)
        most_similar_courses = most_similar_courses[:k]

        recommended_courses = []
        for course_id, similarity in most_similar_courses:
            recommended_courses.append(course_id)

        return recommended_courses
