import glob
import os
from pathlib import Path

import pandas as pd
from jsonlines import jsonlines
from tqdm import tqdm

from utils.cache import cache
from utils.evaluation_utils import metric_max_over_ground_truths, exact_match_score_normalized, \
    exact_match_score
from utils.data import dump_json, load_json, load_jsonl

def get_relations():
    relations = []
    paths = glob.glob("../data/results/all/*.jsonl")

    for path in paths:
        path = path.replace(".jsonl", "")
        relation = path.split('/')[-1]
        if relation.endswith('reranked'): continue
        relations.append(relation)
    return relations


def split_results(relations):
    def split(relation, dataset, env, r):
        print('Splitting',relation, dataset, env)
        dataset_path = f'../data/annotated_datasets/datasets.augmented.jsonl'
        in_path = f'../data/results/all/{relation}.jsonl'
        out_path = f'../data/results/{dataset}-{env}/{relation}'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = f'{out_path}/results.retrieved.jsonl'
        if os.path.exists(out_path):
            return

        with jsonlines.open(out_path, mode='w') as jsonl_writer, jsonlines.open(in_path) as results_reader, jsonlines.open(dataset_path) as dataset_reader:
            for i, (qa, qa_dataset) in enumerate(zip(results_reader, dataset_reader)):
                if i in r:
                    qa['input_qa'] = qa_dataset
                    jsonl_writer.write(qa)

    list = [
        ('nq-open', 'dev', range(0, 8757)),
        ('nq-open', 'test', range(8757, 12367)),
        ('triviaqa', 'dev', range(12367, 21204)),
        ('triviaqa', 'test', range(21204, 32517)),
    ]

    for relation in relations:
        for (dataset, env, r) in list:
            split(relation, dataset, env, r)


def augment_qa_em_scores(qa):
    for k in [1, 5, 10, 50]:
        em = any([
            metric_max_over_ground_truths(exact_match_score_normalized, pred_answer['answer'][0], qa['answer'] + qa['answer_alias'],
                                          qa['question'])
            for pred_answer in qa['retrieved_qas'][:k]
        ])
        qa['retrieved_qa_score'] = qa['retrieved_qas'][0]['score']
        qa[f'em_n_{k}'] = 1 if em else 0
        em = any([
            metric_max_over_ground_truths(exact_match_score, pred_answer['answer'][0], qa['answer'],
                                          qa['question'])
            for pred_answer in qa['retrieved_qas'][:k]
        ])
        qa[f'em_{k}'] = 1 if em else 0

    return qa


@cache('get_qas_em_2')
def get_qas_em(relation, dataset, env):
    print(relation, dataset, env)
    path = f'../data/results/{dataset}-{env}/{relation}'
    path_in = f'{path}/results.retrieved.jsonl'
    qas = []

    path_dataset = f'../data/annotated_datasets/{dataset}.{env}'
    in_path_dataset_original = f'{path_dataset}.jsonl'
    in_path_dataset_augmented = f'{path_dataset}.augmented.jsonl'
    qas_results = load_jsonl(path_in)
    qas_original = load_jsonl(in_path_dataset_original)
    qas_augmented = load_jsonl(in_path_dataset_augmented)

    for i, (qa_augmented, qa_original, qa_result ) in enumerate(tqdm(zip(qas_augmented, qas_original, qas_results))):
        qa = qa_augmented
        answer_original = qa_original['answer']
        answer_all = qa['answer']
        qa['answer_alias'] = list(set(answer_all) - set(answer_original))
        qa['answer'] = answer_original
        qa['retrieved_qas'] = qa_result['retrieved_qas']
        qa = augment_qa_em_scores(qa)
        qas.append(qa)
    return qas


def evaluate(relation, dataset, env):
    print(relation, dataset, env)
    evaluation = {'name': relation}
    qas = get_qas_em(relation, dataset, env)
    path = f'../data/results/{dataset}-{env}/{relation}'

    for k in [1, 5, 10, 50]:
        em = sum([qa[f'em_{k}'] for qa in qas]) / len(qas) * 100
        evaluation[f'em_{k}'] = f'{em:.2f}'

    for k in [1, 5, 10, 50]:
        em = sum([qa[f'em_n_{k}'] for qa in qas]) / len(qas) * 100
        evaluation[f'em_n_{k}'] = f'{em:.2f}'

    print(evaluation)
    dump_json(evaluation, f'{path}/evaluation.json')
    return evaluation


def format_qa(qa):
    out = {
        'id': qa['id'],
        'question': qa['question'],
        'answer': ', '.join(qa['answer']),
        'retrieved_answer': ', '.join(qa['retrieved_qas'][0]['answer']),
        'retrieved_answer_baseline': ', '.join(qa['retrieved_qas_baseline'][0]['answer']),
        'retrieved_qas': '\n'.join(
            [f"{i['question']} ({i['score']:.2f})\n{', '.join(i['answer'])}" for i in qa['retrieved_qas'][:5]]),
        'retrieved_qas_baseline': '\n'.join(
            [f"{i['question']} ({i['score']:.2f})\n{', '.join(i['answer'])}" for i in
             qa['retrieved_qas_baseline'][:5]]),
        'retrieved_qa_score': f"{qa['retrieved_qa_score']:.2f}",
        'wikidata_relations': ', '.join(
            [f'{rel} ({count})' for rel, count in qa.get('wikidata_relations', {}).items()]),
        'wikidata_relation_cardinality': min(qa.get('wikidata_relations').values()) if qa.get(
            'wikidata_relations') else 0,
        'dbpedia_relations': ', '.join([f'{rel} ({count})' for rel, count in qa.get('dbpedia_relations', {}).items()]),
        'dbpedia_relation_cardinality': min(qa.get('dbpedia_relations').values()) if qa.get('dbpedia_relations') else 0,
        'wikidata_entity': qa.get('wikidata_entity', ''),
        'wikidata_entity_id': qa.get('wikidata_entity_id', ''),
        'dbpedia_entity_id': qa.get('dbpedia_entity_id', ''),
        'wikipedia_url': qa.get('wikipedia_url', ''),
    }

    return out


def hits_and_misses(relation, dataset, env, name=None, kb='wikidata'):
    print(relation, dataset, env)
    qas = get_qas_em(relation, dataset, env)
    baseline_qas = get_qas_em('baseline', dataset, env)

    hits = []
    misses = []
    mistakes = []
    k = 5
    for i, (qa, qa_base) in enumerate(zip(qas, baseline_qas)):
        combined_qa = {'id': i + 1,
                       **qa,
                       'retrieved_qas_baseline': qa_base['retrieved_qas']}
        print(qa_base.get(f'{kb}_relations', {}))
        print(qa_base.get(f'{kb}_relations', {}).get(relation.replace('_el','').replace('_', ' ')))
        if name is None:
            name = relation.replace('_el','').replace('_', ' ')
        if (not qa[f'em_n_{k}']) and (not qa_base[f'em_n_{k}']) \
                and qa_base.get(f'{kb}_relations', {}).get(name):
            mistakes.append(combined_qa)
        if qa[f'em_n_{k}'] and not qa_base[f'em_n_{k}']:
            hits.append(combined_qa)
        elif not qa[f'em_n_{k}'] and qa_base[f'em_n_{k}']:
            misses.append(combined_qa)
    path = f'../data/results/{dataset}-{env}/{relation}'

    hits_f = [format_qa(qa) for qa in hits]
    misses_f = [format_qa(qa) for qa in misses]
    mistaks_f = [format_qa(qa) for qa in mistakes]
    pd.DataFrame(hits_f).to_csv(f'{path}/hits.csv')
    pd.DataFrame(misses_f).to_csv(f'{path}/misses.csv')
    pd.DataFrame(mistaks_f).to_csv(f'{path}/mistakes.csv')
    return hits, misses, mistaks_f


def load_evaluation(relation, dataset, env):
    evaluation = {'name': relation}
    evaluation = {**evaluation, **load_json(f'../data/results/{dataset}-{env}/{relation}/evaluation.json')}
    return evaluation

def evaluate_all(relations):
    for dataset in ['nq-open', 'triviaqa']:
        for env in ['dev', 'test']:
            for relation in relations:
                evaluation = evaluate(relation, dataset, env)

def generate_evaluation_table(relations):
    for dataset in ['nq-open', 'triviaqa']:
        for env in ['dev', 'test']:
            # for relation in relations:
            #     hits, misses = hits_and_misses(relation, dataset, env)
            evaluations = []
            baseline = load_evaluation('baseline', dataset, env)
            for relation in relations:
                evaluation = load_evaluation(relation, dataset, env)
                for k in [1, 5, 10, 50]:
                    delta = float(evaluation[f'em_n_{k}']) - float(baseline[f'em_n_{k}'])
                    evaluation[f'em_n_{k}_delta'] = float(f'{delta:.2f}')
                evaluations.append(evaluation)
            pd.DataFrame(evaluations).sort_values('em_n_5_delta', ascending=False).to_csv(f'../data/results/{dataset}-{env}/evaluations.csv')

if __name__ == '__main__':
    relations = get_relations()
    split_results(relations)

    evaluate_all(relations)
    generate_evaluation_table(relations)
    # hits, misses, mistakes = hits_and_misses('performer', 'nq-open', 'dev', 'performer')
