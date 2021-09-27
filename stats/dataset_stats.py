from copy import copy

from jsonlines import jsonlines
import pandas as pd


def load_data(path):
    with jsonlines.open(path) as reader:
        qas = [qa for qa in reader]

    qas_expanded = []
    for qa in qas:
        qa['kb_relation'] = None
        qa['kb_relation_cardinality'] = 0
        qa['kb_relation_source'] = None

        if (not qa.get('wikidata_relations')) and (not qa.get('dbpedia_relations')):
            qas_expanded.append(qa)
            continue
        for source in ['wikidata', 'dbpedia']:
            relations = qa.get(f'{source}_relations', {})
            if relations.get('name'):
                relations = {'name': relations['name']}
            for relation, cardinality in relations.items():
                qac = copy(qa)
                qac['kb_relation'] = relation
                qac['kb_relation_cardinality'] = cardinality
                qac['kb_relation_source'] = source
                qas_expanded.append(qac)

    df = pd.DataFrame.from_records(qas_expanded)
    df['wikidata_entity_id'] = df['wikidata_entity_id'].str.extract(r'http:\/\/www.wikidata.org\/entity\/(Q.+)')
    return df


def print_dataset_stats(dataset, env, kb):
    path = f'../data/results/{dataset}-{env}/baseline/{dataset}-{env}.baseline.jsonl'
    df = load_data(path)
    df_f = df
    df_w = df_f[(df_f['kb_relation_source'] == kb)]
    for name, group in df_w.groupby('kb_relation'):
        rows = len(group)
        if rows < 5: continue
        correct = group['em_n_5'].sum()
        incorrect = rows - correct
        em = (correct / rows) * 100
        avg_card = group['kb_relation_cardinality'].mean()
        print(f"{name}\t{rows}\t{incorrect}\t{em:.2f}\t{avg_card:.1f}")


if __name__ == '__main__':
    dataset = 'nq-open'  # triviaqa
    env = 'dev'
    kb = 'dbpedia'  # wikidata

    print_dataset_stats(dataset, env, kb)
