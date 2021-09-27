import functools
import re
import string
from typing import Union, List
from datetime import datetime

from regex import regex
import number_parser
import pendulum

# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth, question=""):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def normalize_date(string):
    try:
        date = datetime.fromisoformat(string.replace('Z', '+00:00'))
        return date.strftime("%B %-d, %Y")
    except:
        return string


year_pattern = regex.compile('\d{4}$')


def potential_year(string):
    return year_pattern.search(string)


@functools.lru_cache(maxsize=None)
def try_parsing_date(text, formats):
    for fmt in formats:
        try:
            return pendulum.from_format(text, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found')


@functools.lru_cache(maxsize=None)
def exact_match_date(prediction, ground_truth):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|in|on)\b', ' ', text).strip()

    pred = remove_articles(prediction)
    gt = remove_articles(ground_truth)
    if pred == gt:
        return True

    if not (potential_year(pred) and potential_year(ground_truth)):  # speed up
        return False

    try:
        prediction_date = try_parsing_date(pred, ('D MMMM YYYY', 'MMMM D, YYYY', 'MMMM D YYYY', 'MMMM YYYY', 'YYYY'))
    except ValueError as e:
        return False

    try:
        ground_truth_date = try_parsing_date(gt, ('D MMMM YYYY', 'MMMM D, YYYY', 'MMMM D YYYY',))
        return ground_truth_date.date() == prediction_date.date()
    except ValueError:
        pass

    try:
        ground_truth_date = try_parsing_date(gt, ('MMMM YYYY',))
        return ground_truth_date.month == prediction_date.month and ground_truth_date.year == prediction_date.year
    except ValueError:
        pass

    try:
        ground_truth_date = try_parsing_date(gt, ('YYYY',))
        return ground_truth_date.year == prediction_date.year
    except ValueError:
        return False


@functools.lru_cache(maxsize=None)
def extract_amount(text):
    text = number_parser.parse(text)
    text = re.sub("[^0-9.]", "", text)  # remove all non number characters

    try:
        amount = float(text)
        return amount
    except ValueError:
        return None


def exact_match_amount(prediction, ground_truth):
    prediction_amount = extract_amount(prediction)
    ground_truth_amount = extract_amount(ground_truth)
    return prediction_amount == ground_truth_amount and prediction_amount is not None


@functools.lru_cache(maxsize=None)
def exact_match_score_normalized(prediction, ground_truth, question=""):
    if normalize_answer(prediction) == normalize_answer(ground_truth):
        return True

    if exact_match_date(prediction, ground_truth):
        return True

    if re.match(r'^how (many|much) ', question):
        if exact_match_amount(prediction, ground_truth):
            return True

    return False


def ems_norm(prediction, ground_truths, question=""):
    return max([exact_match_score_normalized(prediction, gt, question) for gt in ground_truths])


def ems(prediction, qa, normalized=False):
    ground_truths = qa['answer']
    question = qa['question']
    if normalized:
        ground_truths = qa['answer'] + qa.get('answer_alias', [])
        return max([exact_match_score_normalized(prediction, gt, question) for gt in ground_truths])
    else:
        return max([exact_match_score(prediction, gt) for gt in ground_truths])


def metric_max_over_ground_truths(metric_fn, predictions: Union[str, List[str]], ground_truths: List[str], question):
    scores_for_ground_truths = []

    if isinstance(predictions, str):
        predictions = [predictions]

    for prediction in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth, question)
            scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)