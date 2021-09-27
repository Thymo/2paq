import csv

from jsonlines import jsonlines
from tqdm import tqdm

from utils.evaluation_utils import metric_max_over_ground_truths, exact_match_score_normalized, \
    exact_match_score

def format_qa(qa):
    out = {
        'id': qa['id'],
        'question': qa['question'],
        'answer': ', '.join(qa['answer_original']),
        'answer_alias': ', '.join(list(set(qa['answer']) - set(qa['answer_original']))),
        'retrieved_answer': ', '.join(qa['retrieved_qas'][0]['answer']),
        'retrieved_qas': '\n'.join(
            [f"{i['question']} ({i['score']:.2f})\n{', '.join(i['answer'])}" for i in qa['retrieved_qas'][:5]]),
        'retrieved_qa_score': f"{qa['retrieved_qa_score']:.2f}",
        'wikidata_entity': qa.get('wikidata_entity', ''),
        'wikidata_relations': ', '.join([f'{rel} ({count})' for rel, count in qa.get('wikidata_relations', {}).items()]),
        'wikidata_relation_cardinality': min(qa.get('wikidata_relations').values()) if qa.get('wikidata_relations') else 0,
        'dbpedia_relations': ', '.join([f'{rel} ({count})' for rel, count in qa.get('dbpedia_relations', {}).items()]),
        'dbpedia_relation_cardinality': min(qa.get('dbpedia_relations').values()) if qa.get('dbpedia_relations') else 0,
        'wikidata_entity_id': qa.get('wikidata_entity_id', ''),
        'dbpedia_entity_id': qa.get('dbpedia_entity_id', ''),
        'wikipedia_url': qa.get('wikipedia_url', ''),
        'em_1': qa['em_1'],
        'em_n_1': qa['em_n_1'],
        'em_5': qa['em_5'],
        'em_n_5': qa['em_n_5'],
        'em_10': qa['em_10'],
        'em_n_10': qa['em_n_10'],
        'em_50': qa['em_50'],
        'em_n_50': qa['em_n_50']
    }

    return out

def augment_qa_em_scores(qa):
    for k in [1, 5, 10, 50]:
        em = any([
            metric_max_over_ground_truths(exact_match_score_normalized, pred_answer['answer'][0], qa['answer'], qa['question'])
            for pred_answer in qa['retrieved_qas'][:k]
        ])
        qa['retrieved_qa_score'] = qa['retrieved_qas'][0]['score']
        qa[f'em_n_{k}'] = 1 if em else 0
        em = any([
            metric_max_over_ground_truths(exact_match_score, pred_answer['answer'][0], qa['answer_original'], qa['question'])
            for pred_answer in qa['retrieved_qas'][:k]
        ])
        qa[f'em_{k}'] = 1 if em else 0

    return qa

def evaluate(dataset, env):
    path = f'../data/results/{dataset}-{env}/baseline'
    in_path = f'{path}/results.retrieved.jsonl'
    out_path = f'{path}/{dataset}-{env}.baseline.jsonl'
    out_path_tsv = f'{path}/{dataset}-{env}.baseline.tsv'

    path_dataset = f'../data/annotated_datasets/{dataset}.{env}'
    in_path_dataset_original = f'{path_dataset}.jsonl'
    in_path_dataset_augmented = f'{path_dataset}.augmented.jsonl'

    with jsonlines.open(out_path, mode='w') as jsonl_writer, open(out_path_tsv, mode='w') as out_file_tsv, jsonlines.open(in_path) as results_reader, jsonlines.open(in_path_dataset_original) as original_dataset_reader, jsonlines.open(in_path_dataset_augmented) as augmented_dataset_reader:
        fieldnames = ['id', 'question', 'answer', 'answer_alias', 'retrieved_answer', 'retrieved_qas', 'retrieved_qa_score',
                      'wikidata_entity', 'wikidata_relations', 'wikidata_relation_cardinality', 'dbpedia_relations',
                      'dbpedia_relation_cardinality', 'wikidata_entity_id', 'dbpedia_entity_id', 'wikipedia_url',
                      'em_1', 'em_n_1', 'em_5', 'em_n_5', 'em_10', 'em_n_10', 'em_50', 'em_n_50']
        tsv_writer = csv.DictWriter(out_file_tsv, delimiter='\t', fieldnames=fieldnames)
        tsv_writer.writeheader()

        for i, (qa_original, qa_augmented, qa_result) in enumerate(tqdm(zip(original_dataset_reader, augmented_dataset_reader, results_reader))):
            qa = qa_augmented
            qa['answer_original'] = qa_original['answer']
            qa['id'] = i + 1
            qa['retrieved_qas'] = qa_result['retrieved_qas']
            qa = augment_qa_em_scores(qa)
            formatted_qa = format_qa(qa)
            tsv_writer.writerow(formatted_qa)
            jsonl_writer.write(qa)

if __name__=='__main__':
    for dataset in ['nq-open', 'triviaqa']:
        for env in ['dev', 'test']:
            evaluate(dataset, env)
