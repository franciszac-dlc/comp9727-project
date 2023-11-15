from collections import defaultdict
from itertools import combinations
from abc import ABC, abstractmethod
from typing import Callable, Any

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


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


class ItemBasedCF:
    def __init__(self, course_set: pd.DataFrame):
        self.course_similarity_matrix = None

    def fit(self, training_data: pd.DataFrame):
        # Creating a user-item matrix
        user_item_matrix = training_data.pivot_table(index='reviewers', columns='course_id', values='rating')
        user_item_matrix_filled = user_item_matrix.fillna(0)

        # using cosine similarity for courses
        similarity_matrix = cosine_similarity(user_item_matrix_filled.T)
        np.fill_diagonal(similarity_matrix, 0)
        self.course_similarity_matrix = pd.DataFrame(similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    def recommend(self, prev_courses: tuple, k: int = 5):
        return self.generate_recommendations(prev_courses, k)

    def generate_recommendations(self, prev_courses, k):
        # Ensure prev_courses is a list...
        prev_courses = list(prev_courses) 

        # Safely calculate mean similarities
        try:
            course_similarities = self.course_similarity_matrix.loc[prev_courses]
            recommended_courses = course_similarities.mean(axis=0).sort_values(ascending=False)
        except KeyError:  # Handling case where prev_courses are not in the similarity matrix
            return []

        # Excluding courses already taken by the user
        recommended_courses = recommended_courses[~recommended_courses.index.isin(prev_courses)]

        return recommended_courses.head(k).index.tolist()
    
    def predict_all_ratings(self, test_data: pd.DataFrame):
        results_df = pd.DataFrame()

        for user_id in tqdm(test_data['reviewers'].unique()):
            user_history = test_data[test_data['reviewers'] == user_id]['course_id'].values

            # Calculate mean similarity for each course
            course_similarities = self.course_similarity_matrix.loc[user_history]
            predicted_ratings = course_similarities.mean(axis=0).sort_values(ascending=False)

            # courses the user has already taken
            predicted_ratings = predicted_ratings[~predicted_ratings.index.isin(user_history)]

            user_df = pd.DataFrame({
                'user_id': user_id,
                'course_id': predicted_ratings.index,
                'predicted_rating': predicted_ratings.values
            })

            results_df = pd.concat([results_df, user_df])

        return results_df.reset_index(drop=True)


class UserBasedCF(RecommenderModel):
    def __init__(self):
        self.user_similarity_matrix = None
        self.train_ratings = None

    def fit(self, training_data: pd.DataFrame):
        # Store the training data
        self.train_ratings = training_data
        # Create a user-item matrix
        user_item_matrix = self.create_user_item_matrix(training_data)
        # Calculate the similarity matrix
        self.user_similarity_matrix = self.calculate_similarity(user_item_matrix)

    def create_user_item_matrix(self, training_data):
        user_item_matrix = training_data.pivot_table(index='reviewers', columns='course_id', values='rating')
        user_item_matrix = user_item_matrix.fillna(0)
        return user_item_matrix

    def calculate_similarity(self, user_item_matrix):
        similarity_matrix = cosine_similarity(user_item_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        return pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    def recommend(self, user_id, k: int = 5):
        return self.generate_recommendations(user_id, k)

    def generate_recommendations(self, user_id, k):
        # Get the top similar users
        try:
            similar_users = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False).head(k).index
        except KeyError:
            return []

        # Filter the training data to only include ratings from similar users
        similar_users_ratings = self.train_ratings[self.train_ratings['reviewers'].isin(similar_users)]

        # Aggregate these ratings to get an average rating for each course
        aggregated_ratings = similar_users_ratings.groupby('course_id')['rating'].mean().sort_values(ascending=False)

        # Get top-k recommendations (courses with the highest average rating)
        top_k_courses = aggregated_ratings.head(k).index.tolist()
        return top_k_courses
    
    def predict_all_ratings(self, test_data: pd.DataFrame, k: int = 5):
        results_df = pd.DataFrame()

        for user_id in tqdm(test_data['reviewers'].unique()):
            try:
                similar_users = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False).head(k).index
            except KeyError:
                continue

            # ratings from similar users
            similar_users_ratings = self.train_ratings[self.train_ratings['reviewers'].isin(similar_users)]
            
            # Aggregate to predict scores
            predicted_ratings = similar_users_ratings.groupby('course_id')['rating'].mean().sort_values(ascending=False)

            user_rated_courses = test_data[test_data['reviewers'] == user_id]['course_id']
            predicted_ratings = predicted_ratings[~predicted_ratings.index.isin(user_rated_courses)]

            user_df = pd.DataFrame({
                'user_id': user_id,
                'course_id': predicted_ratings.index,
                'predicted_rating': predicted_ratings.values
            })

            results_df = pd.concat([results_df, user_df])

        return results_df.reset_index(drop=True)


class Content_Based(RecommenderModel):
    def __init__(self, cutoff: int, course_set: pd.DataFrame, Vectorizer, n_features):
        """
        Initializes the Content_Based recommender model.

        Parameters:
        cutoff (int): The cutoff value for the recommender.
        course_set (pd.DataFrame): The set of courses available for recommendation.
        Vectorizer: The vectorizer to be used for transforming course descriptions and skills into vectors.
        n_features (int): The maximum number of features to be used by the vectorizer.
        """
        self.cutoff = cutoff
        self.course_set = course_set

        # Move to the fit function?
        self.vectorizer = Vectorizer(max_features=n_features)
        self.vectorizer.fit(self.course_set["description"] + self.course_set["skills"])

        self.course_vectors = {}
        for id, row in self.course_set.iterrows():
            self.course_vectors[id] = self.vectorizer.transform(
                [row["description"] + row["skills"]]
            )

    def recommend(self, prev_courses: tuple, k: int = 5):
        """
        Recommends courses based on the courses previously taken by the user.

        Parameters:
        prev_courses (tuple): The courses previously taken by the user.
        k (int): The number of courses to recommend. Default is 5.

        Returns:
        list: The IDs of the recommended courses.
        """
        most_similar_courses = self.find_most_similar_courses(prev_courses)[:k]

        recommended_courses = []
        for course_id, similarity in most_similar_courses:
            recommended_courses.append(course_id)

        return recommended_courses

    def find_most_similar_courses(self, prev_courses):
        """
        Finds the most similar courses to the user's previously reviewed courses.

        Parameters:
        prev_courses (tuple): The courses previously reviewed by the user.

        Returns:
        list: A sorted list of tuples where each tuple contains a course ID and its similarity score with the user_vector.
        """

        user_reviews_skills = self.course_set[self.course_set.index in prev_courses][
            "skills"
        ]
        user_reviews_description = self.course_set[
            self.course_set.index in prev_courses
        ]["description"]
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

        return most_similar_courses
