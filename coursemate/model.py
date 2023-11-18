from collections import defaultdict
from itertools import combinations
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Union

from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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

    def generate_scores(self, user: str, prev_courses: tuple):
        prev_courses_ndx = tuple(self.course_to_index[c] for c in prev_courses)
        antecedents = AssociationMiningModel.generate_antecedents(prev_courses_ndx)

        _candidate_set = (
            self.frequent_subseqs[
                self.frequent_subseqs["antecedent"].isin(antecedents)
            ][["consequent", "confidence"]]
            .drop_duplicates(subset=["consequent"])
        )[["consequent", "confidence"]]

        _candidate_set["course"] = _candidate_set["consequent"].apply(lambda x: self.index_to_course[x[0]])
        _candidate_set["user"] = user

        return _candidate_set[['user', 'course', 'confidence']]


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


class ContentBasedModel(RecommenderModel):
    def __init__(
        self,
        course_set: pd.DataFrame,
    ):
        """
        Initializes the Content Based recommender model.

        Parameters:
        course_set (pd.DataFrame): The set of courses available for recommendation.
        """
        self.course_set = course_set
        
        

    def fit(self, Vectorizer: Union[TfidfVectorizer, CountVectorizer] = TfidfVectorizer,
        n_features: int = 10000, categories: List[str]=['data','description']):
        """
        Fit the vectorizer on the combined text of specified columns in the course_set DataFrame.
        Generate course vectors using the fitted vectorizer and store them in the course_vectors attribute.

        Parameters:
        Vectorizer (Union[TfidfVectorizer, CountVectorizer]): The vectorizer to be used for transforming course descriptions and skills into vectors.
        n_features (int): The maximum number of features to be used by the vectorizer.
        categories (List[str]): The list of columns to be combined for generating course vectors.

        Returns:
        None
        """
        
        self.categories = categories
        combined_series = self.course_set[self.categories].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        self.vectorizer = Vectorizer(max_features=n_features)
        self.vectorizer.fit(combined_series)
        
        self.course_vectors = {}
        for id, row in self.course_set.iterrows():
            combined_string = ' '.join(row[field] for field in self.categories)
            self.course_vectors[id] = self.vectorizer.transform([combined_string])
 

    def preprocess(self):
        """
        Preprocess the 'skills' and 'description' columns in the course_set DataFrame.

        This method applies text processing techniques such as lemmatization, removal of stop words, and special character cleanup.
        The processed values are then stored back in the respective columns of the course_set DataFrame.

        Returns:
        None
        """
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def process_skills(skill_text):
            skills = set(skill_text.replace(')','').replace('(','').replace('-',' ').lower().split())
            return ' '.join(skills)

        def process_description(description):
            description = description.lower()
            description = re.sub(r'[^\w\s]', '', description)
            tokens = word_tokenize(description)
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)        

        self.course_set.loc[:, 'skills'] = self.course_set['skills'].apply(process_skills)
        self.course_set.loc[:, 'description'] = self.course_set['description'].apply(process_description)

    def recommend(self, prev_courses: tuple, k: int = 5):
        """
        Recommends courses based on the courses previously taken by the user.

        Parameters:
        prev_courses (tuple): The courses previously taken by the user.
        k (int): The number of courses to recommend. Default is 5.

        Returns:
        list: The IDs of the recommended courses.
        """


        selected_rows = self.course_set[self.course_set.index.isin(prev_courses)]
        user_reviews_combined = selected_rows[self.categories].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        #user_reviews_skills = self.course_set[self.course_set.index.isin(prev_courses)]["skills"]
        #user_reviews_description = self.course_set[self.course_set.index.isin(prev_courses)]["description"]
        #user_reviews_combined = user_reviews_skills + " " + user_reviews_description
        #print(type(user_reviews_skills + " " + user_reviews_description))
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
        return most_similar_courses

    def ContentBasedModel_gridsearch(self,x:pd.DataFrame, y:pd.DataFrame,vectorizers:List[Union[TfidfVectorizer, CountVectorizer]],n_users:int = 250, n_features:List[int]=[100,1000,10000],categories: List[List[str]]=[['skills'],['description'], ['skills','description']],k_list:List[int]=[5,10]):
        """
        Perform grid search for Content-Based Recommender Model with different configurations.

        Parameters:
        x (pd.DataFrame): Training data.
        y (pd.DataFrame): Target data.
        vectorizers (List[Union[TfidfVectorizer, CountVectorizer]]): List of vectorizers to be used in the grid search.
        n_users (int): Number of users to consider in the grid search.
        n_features (List[int]): List of numbers of features to consider in the grid search.
        categories (List[List[str]]): List of column combinations to consider in the grid search.
        k_list (List[int]): List of values for k in the grid search.

        Returns:
        List[Dict]: List of dictionaries containing grid search results for different configurations.
        """
        self.preprocess()

        unique_reviewers_X = x['reviewers'].unique()
        unique_reviewers_y = y['reviewers'].unique()
        results = []
        for n in n_features:
            for vectorizer in vectorizers:
                for data in categories:
                    self.fit(vectorizer,n,data)

                    statistics = {k: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0} for k in k_list}
                    hits = {k: 0 for k in k_list}
                    count = 0

                    for user in unique_reviewers_X:
                        if user in unique_reviewers_y:
                            count += 1
                            user_profile = tuple(x[x['reviewers'] == user]['course_id'])
                            target = y[y['reviewers'] == user]['course_id']

                            recommendations = self.recommend(user_profile,None)
                            recommendation_ids = [recommendation[0] for recommendation in recommendations]
                            
                            # Statistics gathering 
                            for k in k_list:
                                statistics[k]['true_positives'] += len(set(target) & set(recommendation_ids[:k]))
                                statistics[k]['false_positives'] += len(set(recommendation_ids[:k]) - set(target))
                                statistics[k]['false_negatives'] += len(set(target) - set(recommendation_ids[:k]))
                                if len(set(target) & set(recommendation_ids[:k])) > 0:
                                    hits[k] += 1
                        if count == n_users:
                            break

                    # Metric calculation
                    metrics_per_vectorizer = {'Vectorizer': vectorizer.__name__,
                                            'n_features': n,
                                            'categories': data,
                                            'statistics': statistics,
                                            'hits': hits}

                    results.append(metrics_per_vectorizer)
                    # Print or return the results for further analysis
        # for result in results:
        #     print(f"Vectorizer: {result['Vectorizer']}, n_features: {result['n_features']}, categories: {result['categories']}")
        #     for k in k_list:
        #         precision = result['statistics'][k]['true_positives'] / (
        #                 result['statistics'][k]['true_positives'] + result['statistics'][k]['false_positives']) \
        #             if (result['statistics'][k]['true_positives'] + result['statistics'][k]['false_positives']) > 0 else 0
        #         recall = result['statistics'][k]['true_positives'] / (
        #                 result['statistics'][k]['true_positives'] + result['statistics'][k]['false_negatives']) \
        #             if (result['statistics'][k]['true_positives'] + result['statistics'][k]['false_negatives']) > 0 else 0

        #         f1_score = 2 * (precision * recall) / (precision + recall) \
        #             if (precision + recall) > 0 else 0

        #         print(f"For k={k}:")
        #         print(f"Hitrate:{(result['hits'][k] / count):.3f}\tF1 Score:{f1_score:.3f}\tPrecision:{precision:.3f}")
        return results

        