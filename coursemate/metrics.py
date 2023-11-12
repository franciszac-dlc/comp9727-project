from tqdm import tqdm
from typing import Iterable, Callable


def calculate_hit_rate_from_next_course_sequences(
        X: Iterable, y: Iterable, recommender_func: Callable, k: int = 5
    ):
    hits = 0
    reqs = 0
    for (user, prev_courses), next_course in tqdm(zip(X, y), total=len(X)):
        reqs += 1
        if next_course in recommender_func(prev_courses, k):
            hits += 1
    return hits / reqs


def calculate_precision_from_next_course_sets(
        X: Iterable, y: Iterable, recommender_func: Callable, k: int = 5
    ):
    precs = 0
    reqs = 0
    for (user, prev_courses_dict), next_courses_dict in tqdm(zip(X, y), total=len(X)):
        reqs += 1
        prev_courses = set(prev_courses_dict.keys())
        next_courses = set(next_courses_dict.keys())
        pred_courses = set(recommender_func(prev_courses, k))
        precs += (len(pred_courses & next_courses) / max(len(pred_courses), 1))
    return precs / reqs