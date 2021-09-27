import argparse
import csv
import os
import pickle
from collections import Counter
from retrying import retry
import itertools
from urllib.parse import quote

from SPARQLWrapper import SPARQLWrapper2
from jsonlines import jsonlines
from tqdm import tqdm

from utils.cache import cache
from utils.evaluation_utils import ems, normalize_date, unique
from genre.trie import Trie
from genre.hf_model import GENRE
from utils.utils import sort_dict_by_val
from utils.wikidata import augment_entity

parser = argparse.ArgumentParser("Dataset augmenter")
parser.add_argument('--dataset', default='nq-open', type=str, choices=['nq-open', 'triviaqa'])
parser.add_argument('--env', default='dev', type=str, choices=['dev', 'test', 'train'])

args = parser.parse_args()
print(args)
dataset = args.dataset
env = args.env

path = f'../data/annotated_datasets/{dataset}.{env}'
in_path = f'{path}.normalized.jsonl'
out_path_tsv = f'{path}.augmented.tsv'
out_path = f'{path}.augmented.jsonl'

assert os.path.exists(in_path)
assert not os.path.exists(out_path)

with open("../data/genre/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))  # load the prefix tree (trie)

model = GENRE.from_pretrained("../data/genre/hf_wikipage_retrieval").eval()
sparql_dbpedia = SPARQLWrapper2("https://dbpedia.org/sparql")


@cache('dbpedia_query')
@retry(wait_exponential_multiplier=500, stop_max_attempt_number=5, wait_exponential_max=15000)
def dbpedia_query(query):
    try:
        sparql_dbpedia.setQuery(query)
        results = sparql_dbpedia.query().bindings
        return results
    except Exception as err:
        print(err)
        raise err


@cache('link_entity_genre_question')
def link_entity_genre_question(question, threshold=-1.75):
    sentences = [question + "?"]
    entities = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    )
    print(entities)
    if len(entities) == 0:
        return None

    entity = entities[0][0]
    if entity['logprob'] >= threshold:
        return entity['text']

    return None


def augment_entity_dbpedia(name):
    try:
        results = dbpedia_query(f"""
        select distinct ?item ?property ?propertyLabel ?object ?objectLabel {{
          ?item rdfs:label "{name}"@en .
          ?item ?property ?object 
          optional {{
            ?property rdfs:label ?propertyLabel .
            filter langMatches(lang(?propertyLabel), 'en')
        }}
          optional {{
            ?object rdfs:label ?objectLabel .
            filter langMatches(lang(?objectLabel), 'en')
          }}
          bind(datatype(?object) as ?dt)
          filter(if(isliteral(?object) && !bound(?dt), langMatches(lang(?object),'en'), true))
          }}
          """)

        if len(results) == 0:
            return None, None

        item_id = results[0]['item'].value

        results = [(r['propertyLabel'].value if r.get('propertyLabel') else '',
                    normalize_date(r.get('objectLabel').value if r.get('objectLabel') else r['object'].value),
                    )
                   for r in results]

        results = {key: [object for _, object in group] for key, group in itertools.groupby(results, lambda r: r[0])}
        results = {key: value for key, value in results.items() if len(value) <= 10 and len(key) > 0}

        return results, item_id

    except Exception as err:
        print(err)
        return None, None


def format_qa(qa):
    out = {
        'question': qa['question'],
        'answer': ', '.join(qa['answer']),
        'wikidata_relations': ', '.join(
            [f'{rel} ({count})' for rel, count in qa.get('wikidata_relations', {}).items()]),
        'dbpedia_relations': ', '.join([f'{rel} ({count})' for rel, count in qa.get('dbpedia_relations', {}).items()]),
        'wikidata_entity': qa.get('wikidata_entity', ''),
        'wikidata_entity_id': qa.get('wikidata_entity_id', ''),
        'dbpedia_entity_id': qa.get('dbpedia_entity_id', ''),
        'wikipedia_url': qa.get('wikipedia_url', '')
    }
    return out


def augment_qa(qa):
    question = qa['question']
    qa['answer'] = [answer for answer in qa['answer'] if len(answer.strip()) > 0]
    entity = link_entity_genre_question(question)

    if entity is None:
        return qa

    print(question)
    print(entity)
    qa['wikidata_entity'] = entity
    qa['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{quote(entity)}"

    entity_data, entity_id = augment_entity(entity)
    if entity_data is not None:

        qa['wikidata_entity_id'] = entity_id
        qa['wikidata_relations'] = []
        entity_data.insert(0, ('name', entity, '', ''))

        pred_vals = unique([(predicate, value) for predicate, value, *_, in entity_data])
        relations = [predicate for predicate, _ in pred_vals]

        for predicate, value, qualifier, qualifier_value in entity_data:
            relations.append(f'{predicate}->{qualifier}')
            if ems(value, qa, True):
                print(predicate, value)
                qa['wikidata_relations'].append(predicate)
            if ems(qualifier_value, qa, True):
                print(f'{predicate}->{qualifier}', qualifier_value)
                qa['wikidata_relations'].append(f'{predicate}->{qualifier}')

        counter = Counter(relations)
        qa['wikidata_relations'] = {r: counter[r] for r in qa['wikidata_relations']}
        qa['wikidata_relations'] = sort_dict_by_val(qa['wikidata_relations'])

    entity_data, entity_id = augment_entity_dbpedia(entity)

    if entity_data is not None:
        qa['dbpedia_relations'] = []
        relations = []
        qa['dbpedia_entity_id'] = entity_id
        for predicate, values in entity_data.items():
            for value in values:
                relations.append(predicate)
                if ems(value, qa, True):
                    print(predicate, value)
                    qa['dbpedia_relations'].append(predicate)

        counter = Counter(relations)
        qa['dbpedia_relations'] = {r: counter[r] for r in qa['dbpedia_relations']}
        qa['dbpedia_relations'] = sort_dict_by_val(qa['dbpedia_relations'])
    return qa

def augment_dataset():
    with jsonlines.open(out_path, mode='w') as writer, open(out_path_tsv, mode='w') as out_file, jsonlines.open(in_path,
                                                                                                                mode='r') as reader:
        fieldnames = ['question', 'answer', 'wikidata_relations', 'dbpedia_relations', 'wikidata_entity',
                      'wikidata_entity_id', 'dbpedia_entity_id', 'wikipedia_url']
        tsv_writer = csv.DictWriter(out_file, delimiter='\t', fieldnames=fieldnames)
        tsv_writer.writeheader()

        for i, qa in enumerate(tqdm(reader)):
            qa = augment_qa(qa)
            writer.write(qa)
            formatted_qa = format_qa(qa)
            tsv_writer.writerow(formatted_qa)


if __name__ == '__main__':
    augment_dataset()