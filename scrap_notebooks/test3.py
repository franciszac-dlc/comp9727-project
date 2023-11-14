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


def hit_rate(model, test_set, k=5):
    hits = 0
    total = 0

    for user_id in test_set['reviewers'].unique():
        prev_courses = set(test_set[test_set['reviewers'] == user_id]['course_id'])

        if not prev_courses:
            continue
        
        user_predicted_top_k = set(model.recommend(list(prev_courses), k))
        user_actual_top_k = set(test_set[test_set['reviewers'] == user_id].nlargest(k, 'rating')['course_id'])

        hits += 1 if len(user_predicted_top_k & user_actual_top_k) > 0 else 0
        total += 1
        
        # print(user_predicted_top_k)
        # print(user_actual_top_k)
        
        print(f"Hits: {hits}")

    hit_rate = hits / total if total > 0 else 0
    print(f"Final Hit Rate: {hit_rate}")
    return hit_rate


# cf_model = UserBasedCF()
cf_model = ItemBasedCF(dataset.course_set)


cf_model.fit(dataset.train_ratings)

# print(cf_model.recommend('By Kelvin k', k=10))


hit_rate(cf_model, dataset.test_ratings, k=400)

# Test the model
# precision = precision_at_k(cf_model, dataset.test_ratings, k=10)
# print(f"Precision@5: {precision}")