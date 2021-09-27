import numpy as np

from utils.data import load_json
from utils.evaluation_utils import ems

models = {
    'PAQ-retriever': 'PAQ/test.repaq_retriever.nq.txt',
    'PAQ-reranker': 'PAQ/test.repaq_reranker.nq.txt',
    'fid': 'fid/test.fid.nq.txt',
    'rag': 'rag/test.rag.nq.txt',
    't5': 't5/test.t5.nq.txt',
    'bart': 'bart/test.bart.nq.txt',
}

dataset = 'nq-open'
env = 'test'
path_dataset = f'../data/annotated_datasets/{dataset}.{env}'

refs = load_json('data/nqopen-test.augmented.json')
test_categories = load_json('../data/nq-open/nq_test_full.json')

def load_answers(path):
    answers = []
    with open(f'../data/all_model_preds/{path}', 'r') as f:
        for line in f:
            answers.append(line.strip())
    return answers

def ems_models(models):
    for model, path in models.items():
        preds = load_answers(path)
        em = []
        for pred, qa, category in zip(preds, refs, test_categories):
            em.append(ems(pred, qa, True))
        print(f'{model} {np.mean(em)*100:.2f}')

if __name__=='__main__':
    ems_models(models)