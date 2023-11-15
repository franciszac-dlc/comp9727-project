from tqdm import tqdm
from typing import Iterable, Callable


def calculate_hit_rate_from_next_course_sequences(
        X: Iterable, y: Iterable, recommender_func: Callable, k: int = 5
    ):
    hits = 0
    reqs = 0
    for (user, prev_courses), next_course in tqdm(zip(X, y), total=len(X)):
        reqs += 1
        pred_courses = set(recommender_func(prev_courses, k))
        if len(pred_courses & set(next_course)) > 0:
            hits += 1
    return hits / reqs


def calculate_precision_from_next_course_sequences(
        X: Iterable, y: Iterable, recommender_func: Callable, k: int = 5
    ):
    precs = 0
    reqs = 0
    for (user, prev_courses), next_course in tqdm(zip(X, y), total=len(X)):
        reqs += 1
        pred_courses = set(recommender_func(prev_courses, k))
        precs += (len(pred_courses & set(next_course)) / max(len(pred_courses), 1))
    return precs / reqs


def calculate_f1score_from_next_course_sequences(
        X: Iterable, y: Iterable, recommender_func: Callable, k: int = 5
    ):
    f1scores = 0
    reqs = 0
    for (user, prev_courses), next_course in tqdm(zip(X, y), total=len(X)):
        reqs += 1
        pred_courses = set(recommender_func(prev_courses, k))
        next_courses_set = set(next_course)
        precision = (len(pred_courses & next_courses_set) / max(len(pred_courses), 1))
        recall = (len(pred_courses & next_courses_set) / max(len(next_course), 1))
        if precision + recall > 0:
            f1scores += (2 * (precision * recall) / (precision + recall))
    return f1scores / reqs