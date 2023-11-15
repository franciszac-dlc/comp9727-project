from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Dataset:
    def __init__(self, courses_file, coursedata_file, ratings_file):
        print("Loading Coursera courses...")
        _df_courses = pd.read_csv(courses_file).drop_duplicates()
        _df_coursedata = pd.read_csv(coursedata_file).drop_duplicates()
        self.create_courses_dataframe(_df_courses, _df_coursedata)

        print("Loading Coursera reviews...")
        _df_ratings = pd.read_csv(ratings_file).drop_duplicates()
        self.create_ratings_dataframe(_df_ratings)
        self.create_students_dataframe()


    # Set up the dataframes and overall dataset

    def create_courses_dataframe(self, df_courses, df_coursedata):
        df_coursedata['Course ID'] = df_coursedata['Course URL'].str.split('/').str[-1]

        df_courses_full = df_courses.merge(
            df_coursedata.drop_duplicates('Course ID', keep='first'),
            left_on='course_id',
            right_on='Course ID',
            how='inner'
        )
        df_courses_full = df_courses_full.rename(columns={
            'Difficulty Level': 'difficulty_level',
            'Course Rating': 'rating',
            'Course Description': 'description',
            'Skills': 'skills',
        })
        df_courses_full = df_courses_full[[
            'course_id', 'name', 'institution', 'difficulty_level', 'rating', 'description', 'skills'
        ]]
        df_courses_full.set_index('course_id', inplace=True)

        self.df_courses = df_courses_full


    def create_ratings_dataframe(self, df_ratings):
        self.df_ratings = df_ratings[df_ratings['course_id'].isin(self.df_courses.index)].drop_duplicates()
        self.df_ratings['date_reviews'] = pd.to_datetime(
            self.df_ratings['date_reviews'],
            format="%b %d, %Y"
        )
        self.df_ratings = self.df_ratings.sort_values(by=['reviewers', 'date_reviews']).reset_index(drop=True)


    def create_students_dataframe(self):
        self.df_students = (self.df_ratings.groupby('reviewers').agg({
            'rating': 'mean',
            'course_id': 'count'
        }).rename(columns={'course_id': 'courses', 'rating': 'avg_rating'})
          .sort_values(by='courses', ascending=False))


    def set_interaction_counts(self, lb=3, ub=50):
        print(f"Segmenting out students with less than {lb} or more than {ub} reviews...")
        self.student_set = self.df_students[self.df_students['courses'].between(lb, ub)]
        self.rating_set = self.df_ratings[self.df_ratings['reviewers'].isin(self.student_set.index)]
        self.course_set = self.df_courses[self.df_courses.index.isin(self.rating_set['course_id'])]


    def show_dataset_details(self):
        total_students = len(self.df_students)
        student_count = len(self.student_set)
        total_ratings = len(self.df_ratings)
        rating_count = len(self.rating_set)
        total_courses = len(self.df_courses)
        courses_count = len(self.course_set)

        sparsity = rating_count / (student_count * courses_count)
        duplicates = (1 - (len(self.rating_set.drop_duplicates(['reviewers', 'course_id'])) / len(self.rating_set)))
        
        print(f"{student_count} students, {courses_count} courses, {rating_count} reviews")
        print(f"Sparsity: {100*sparsity:.2f}%")
        print(f"Duplicates: {100*duplicates:.2f}%")


    # Train-test splits

    def set_train_test_split_by_user(self, seed=42):
        """
        Defines a train-test split between users. This allows us to work with
        a small number of users.

        The split is visualized below (training set - o, test set - x)

                        Ratings:
                    1   2   3   4   5
                1   o   o   o   o   o
                2   o   o   o   o   o
        Users:  3   o   o   o   o   o
                4   x   x   x   x   x
                5   x   x   x   x   x

        Sets the following attributes for access:
            - train_students
            - test_students
            - train_ratings
            - test_ratings
        """
        print("Setting the train-test split by user...")
        self.train_students, self.test_students = \
            train_test_split(self.student_set.index, random_state=seed)
        self.train_ratings = self.rating_set[self.rating_set.reviewers.isin(self.train_students)]
        self.test_ratings = self.rating_set[self.rating_set.reviewers.isin(self.test_students)]


    def set_train_test_split_by_ratings(self, ratio=0.8):
        """
        Defines a train-test split between ratings. The first `ratio` of ratings
        chronologically are added to the test set.

        The split is visualized below (training set - o, test set - x)

                        Ratings:
                    1   2   3   4   5
                1   o   o   o   x   x
                2   o   o   o   x   x
        Users:  3   o   o   o   x   x
                4   o   o   o   x   x
                5   o   o   o   x   x

        Sets the following attributes for access:
            - train_ratings
            - test_ratings
        """
        print("Setting the train-test split by rating...")
        training_course_counts = {
            s: int(self.student_set.loc[s, 'courses'] * ratio)
            for s in self.student_set.index
        }

        _train_ratings_ndx = []
        _test_ratings_ndx = []
        for ndx, rating_row in tqdm(self.rating_set[['reviewers']].iterrows()):
            if training_course_counts[rating_row['reviewers']] <= 0:
                _test_ratings_ndx.append(ndx)
            else:
                _train_ratings_ndx.append(ndx)
                training_course_counts[rating_row['reviewers']] -= 1

        self.train_ratings = self.rating_set.loc[pd.Index(_train_ratings_ndx), :]
        self.test_ratings = self.rating_set.loc[pd.Index(_test_ratings_ndx), :]
        
    def set_train_test_split_random(self, ratio=0.8):
        """
        Defines a train-test split randomly. The first `ratio` of ratings
        chronologically are added to the test set.

        The split is visualized below (training set - o, test set - x)

                        Ratings:
                    1   2   3   4   5
                1   o   o   o   x   x
                2   o   o   o   x   x
        Users:  3   o   o   o   x   x
                4   o   o   o   x   x
                5   o   o   o   x   x

        Sets the following attributes for access:
            - train_ratings
            - test_ratings
        """
        print("Setting the train-test split randomly...")
        _train_ratings_ndx, _test_ratings_ndx = train_test_split(self.rating_set.index, train_size=ratio)
        self.train_ratings = self.rating_set.loc[pd.Index(_train_ratings_ndx), :]
        self.test_ratings = self.rating_set.loc[pd.Index(_test_ratings_ndx), :]


    def get_train_test_matrices(self):
        """
        Computes the training and test matrices for rating prediction-related tasks
        This is meant to be ran after running `set_train_test_split_by_ratings`

        Returns
        - training_matrix : np.array(n_students, n_courses)
        - test_matrix     : np.array(n_students, n_courses)
        """
        print("Computing the training and test rating matrix...")
        training_matrix = np.zeros((len(self.student_set), len(self.course_set)), dtype=np.int8)
        for ndx, row in tqdm(self.train_ratings.iterrows()):
            student_ndx = self.student_set.index.get_loc(row['reviewers'])
            course_ndx = self.course_set.index.get_loc(row['course_id'])
            training_matrix[student_ndx, course_ndx] = row['rating']

        test_matrix = training_matrix.copy()
        for ndx, row in tqdm(self.test_ratings.iterrows()):
            student_ndx = self.student_set.index.get_loc(row['reviewers'])
            course_ndx = self.course_set.index.get_loc(row['course_id'])
            test_matrix[student_ndx, course_ndx] = row['rating']

        return training_matrix, test_matrix


    def get_train_matrix_split(self, ratio=0.8):
        """
        Computes the training matrix for rating prediction-related tasks.
        The matrix is split into "input" and "ground truth" for evaluation.
        This is meant to be ran after running `set_train_test_split_by_user`

        The resulting split can be thought of as a train-validation split.

        The split is visualized below (training input - o, ground truth - t)

                        Ratings:
                    1   2   3   4   5
                1   o   o   o   t   t
                2   o   o   o   t   t
        Users:  3   o   o   o   t   t
                4
                5

        Returns
        - training_matrix_features     : np.array(n_train_students, n_courses)
        - training_matrix_ground_truth : np.array(n_train_students, n_courses)
        - df_features                  : pd.DataFrame
        - df_ground_truth              : pd.DataFrame
        """
        print("Computing the training and test rating matrix...")
        train_matrix_ground_truth = np.zeros((len(self.student_set), len(self.course_set)), dtype=np.int8)
        train_matrix_features = train_matrix_ground_truth.copy()
        train_course_counts = {
            s: int(self.student_set.loc[s, 'courses'] * ratio)
            for s in self.train_students
        }

        features_ndx = []
        ground_truth_ndx = []

        for ndx, row in tqdm(self.train_ratings.iterrows()):
            student_ndx = self.student_set.index.get_loc(row['reviewers'])
            course_ndx = self.course_set.index.get_loc(row['course_id'])

            if train_course_counts[row['reviewers']] >= 0:
                train_matrix_features[student_ndx, course_ndx] = row['rating']
                features_ndx.append(ndx)
            else:
                ground_truth_ndx.append(ndx)

            train_course_counts[row['reviewers']] -= 1
            train_matrix_ground_truth[student_ndx, course_ndx] = row['rating']

        df_features = self.train_ratings.loc[features_ndx, :].copy()
        df_ground_truth = self.train_ratings.loc[ground_truth_ndx, :].copy()

        return train_matrix_features, train_matrix_ground_truth, df_features, df_ground_truth


    def get_test_matrix_split(self, ratio=0.5):
        """
        Computes the test matrix for rating prediction-related tasks.
        The matrix is split into "input" and "ground truth" for evaluation.
        This is meant to be ran after running `set_train_test_split_by_user`

        The split is visualized below (training features - o, ground truth - t)

                        Ratings:
                    1   2   3   4   5
                1   o   o   o   o   o
                2   o   o   o   o   o
        Users:  3   o   o   o   o   o
                4   o   o   o   t   t
                5   o   o   o   t   t

        Returns
        - test_matrix_features     : np.array(n_test_students, n_courses)
        - test_matrix_ground_truth : np.array(n_test_students, n_courses)
        - df_features              : pd.DataFrame
        - df_ground_truth          : pd.DataFrame
        """
        print("Computing the test rating matrix split...")
        # test_matrix_ground_truth = np.zeros((len(self.student_set), len(self.course_set)), dtype=np.int8)
        _, test_matrix_ground_truth, df_features, df_ground_truth = self.get_train_matrix_split()
        test_matrix_features = test_matrix_ground_truth.copy()
        test_course_counts = {
            s: int(self.student_set.loc[s, 'courses'] * ratio)
            for s in self.test_students
        }

        features_ndx = []
        ground_truth_ndx = []

        for ndx, row in tqdm(self.test_ratings.iterrows()):
            student_ndx = self.student_set.index.get_loc(row['reviewers'])
            course_ndx = self.course_set.index.get_loc(row['course_id'])

            if test_course_counts[row['reviewers']] >= 0:
                test_matrix_features[student_ndx, course_ndx] = row['rating']
                features_ndx.append(ndx)
            else:
                ground_truth_ndx.append(ndx)

            test_course_counts[row['reviewers']] -= 1
            test_matrix_ground_truth[student_ndx, course_ndx] = row['rating']

        df_features_full = pd.concat([df_features, df_ground_truth, self.test_ratings.loc[features_ndx, :]])
        df_ground_truth_full = self.test_ratings.loc[ground_truth_ndx, :].copy()

        return test_matrix_features, test_matrix_ground_truth, df_features_full, df_ground_truth_full


    def get_train_test_sequence_predictions(self):
        """
        Computes training and test data instances for sequential prediction.
        This is meant to be ran after running `set_train_test_split_by_ratings`

        Returns
        - train_features     : list(tuple( user, tuple(taken_courses) ))
        - train_ground_truth : list(next_course)
        - test_features      : list(tuple( user, tuple(taken_courses) ))
        - test_ground_truth  : list(next_course)
        """
        print("Computing the training and test list of sequences...")
        train_X, train_y = [], []
        current_user = None
        taken_courses = tuple()
        for ndx, row in tqdm(self.train_ratings.iterrows()):
            if current_user != row['reviewers']:
                # New user/sequence, 
                current_user = row['reviewers']
                taken_courses = (row['course_id'],)
            else:
                # Add the previous courses into the train_X list
                train_X.append((current_user, taken_courses))
                train_y.append(row['course_id'])
                taken_courses = (*taken_courses, row['course_id'])

        test_X, test_y = [], []
        current_user = None
        taken_courses = tuple()
        for ndx, row in tqdm(self.test_ratings.iterrows()):
            if current_user != row['reviewers']:
                # New user/sequence, 
                current_user = row['reviewers']
                taken_courses = (row['course_id'],)
            else:
                # Add the previous courses into the train_X list
                test_X.append((current_user, taken_courses))
                test_y.append(row['course_id'])
                taken_courses = (*taken_courses, row['course_id'])

        return train_X, test_X, train_y, test_y


    def get_train_test_next_course_predictions(self, ratio=0.5):
        """
        Computes training and test data instances for next course prediction.

        This is very similar to `get_train_test_sequence_predictions` but 

        Returns
        - train_features     : list(tuple( user, tuple(taken_courses) ) ))
        - test_features      : list(tuple( user, tuple(taken_courses) ))
        - train_ground_truth : list(next_course)
        - test_ground_truth  : list(tuple( next_courses ))
        """
        print("Computing the training and test list of sequences...")
        train_X, train_y = [], []
        current_user = None
        taken_courses = tuple()
        for ndx, row in tqdm(self.train_ratings.iterrows()):
            if current_user != row['reviewers']:
                # New user/sequence, 
                current_user = row['reviewers']
                taken_courses = (row['course_id'],)
            else:
                # Add the previous courses into the train_X list
                train_X.append((current_user, taken_courses))
                train_y.append(row['course_id'])
                taken_courses = (*taken_courses, row['course_id'])

        # For the test set, to match the current splits, it is generated differently
        taken_courses = defaultdict(tuple)  # user -> tuple(courses)
        next_courses = defaultdict(tuple)  # user -> tuple(courses)
        test_course_counts = {
            s: int(self.student_set.loc[s, 'courses'] * ratio)
            for s in self.test_students
        }
        for ndx, row in tqdm(self.test_ratings.iterrows()):
            if test_course_counts[row['reviewers']] >= 0:
                # append to inputs
                taken_courses[row['reviewers']] = (
                    *taken_courses[row['reviewers']],
                    row['course_id']
                )
            else:
                # append to outputs
                next_courses[row['reviewers']] = (
                    *next_courses[row['reviewers']],
                    row['course_id']
                )

            test_course_counts[row['reviewers']] -= 1

        # Generate results
        test_X = list((user, taken_courses[user]) for user in taken_courses.keys())
        test_y = list(next_courses[user] for user in next_courses.keys())

        return train_X, test_X, train_y, test_y