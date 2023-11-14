import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
from coursemate.dataset import Dataset
from coursemate.model import UserBasedCF, ItemBasedCF

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

directory = r'C:\Users\jerem\OneDrive - UNSW\COMP9727'

dataset = Dataset(directory+'\data\Coursera_courses.csv', directory+'\data\Coursera.csv', directory+'\data\Coursera_reviews.csv')
dataset.set_interaction_counts(3, 50)
dataset.show_dataset_details()


dataset.set_train_test_split_random()

# def precision_at_k(model, test_set, k=5):
#     hits = 0
#     total = 0

#     for user_id in test_set['reviewers'].unique():
#         # Check if the user_id exists in the user similarity matrix
#         if user_id not in model.user_similarity_matrix.index:
#             continue  # Skip this user if not in the training set

#         user_actual = test_set[test_set['reviewers'] == user_id]
#         user_actual_top_k = user_actual.nlargest(k, 'rating')['course_id']
#         user_predicted_top_k = model.recommend(user_id, k)
        
#         # print(f"Actual: {user_actual_top_k.values}")
#         # print(f"Predicted: {user_predicted_top_k}")

#         hits += len(set(user_predicted_top_k) & set(user_actual_top_k))
#         total += k
        
#         print(f"Precision@5: {hits / total if total > 0 else 0}")

#     return hits / total if total > 0 else 0

def hit_rate(model, test_set, k=5):
    hits = 0
    total = 0

    for user_id in test_set['reviewers'].unique():
        # Check if the user_id exists in the user similarity matrix
        if user_id not in model.user_similarity_matrix.index:
            continue  # Skip this user if not in the training set

        user_actual = test_set[test_set['reviewers'] == user_id]
        user_actual_top_k = user_actual.nlargest(k, 'rating')['course_id']
        user_predicted_top_k = model.recommend(user_id, k)

        hits += 1 if len(set(user_predicted_top_k) & set(user_actual_top_k)) > 0 else 0
        total += 1
        
        print(f"Hit Rate: {hits / total if total > 0 else 0}")

    return hits / total if total > 0 else 0




cf_model = UserBasedCF()
# cf_model = ItemBasedCF(dataset.course_set)


cf_model.fit(dataset.train_ratings)


hit_rate(cf_model, dataset.test_ratings, k=10)

# Test the model
# precision = precision_at_k(cf_model, dataset.test_ratings, k=10)
# print(f"Precision@5: {precision}")