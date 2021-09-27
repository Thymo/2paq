import argparse
import re
from jsonlines import jsonlines
import wikipedia
from SPARQLWrapper import SPARQLWrapper2
import pickle
from retrying import retry

from utils.cache import cache

from tqdm import tqdm

from genre.trie import Trie
from genre.hf_model import GENRE
from utils.utils import unique

parser = argparse.ArgumentParser("Normalizing datasets")
parser.add_argument('--dataset', default='nq-open', type=str, choices=['nq-open', 'triviaqa'])
parser.add_argument('--env', default='dev', type=str, choices=['dev', 'test', 'train'])

args = parser.parse_args()
print(args)

sparql = SPARQLWrapper2("https://query.wikidata.org/sparql")

# load the prefix tree (trie)
with open("../data/genre/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

model = GENRE.from_pretrained("../data/genre/hf_entity_disambiguation_blink").eval()


@cache('wikipedia')
def get_wikipedia_page(name):
    return wikipedia.page(name, auto_suggest=False)


@cache('wikipedia_search')
def wikipedia_search(query, results=1):
    return wikipedia.search(query, results=results)


def link_entity_wikipedia_search(question, answer):
    results = wikipedia_search(answer)
    if len(results) > 0:
        return str(results[0])
    return None


@cache('link_entity_genre')
def link_entity_genre(question, answer, threshold=-1.5):
    sentences = [f'{question}? [START_ENT] {answer} [END_ENT]']
    entities = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    )
    if len(entities) == 0:
        return None

    entity = entities[0][0]
    if entity['logprob'] > threshold:
        return entity['text']

    return None


def normalize_qa(qa, link_entity_fn=link_entity_genre):
    normalized_answers = []
    for answer in qa['answer']:
        normalized_answers.append(answer)

        wiki_entity_name = link_entity_fn(qa['question'], answer)
        if wiki_entity_name is None:
            wiki_entity_name = link_entity_wikipedia_search(qa['question'], answer)

        if wiki_entity_name is not None:
            normalized_answers += augment_entity_name(wiki_entity_name)

    qa['answer'] = unique(normalized_answers)
    qa['answer'] = [a.strip() for a in qa['answer']]  # filter out empty answers
    qa['answer'] = [a for a in qa['answer'] if a]
    return qa




@cache('wikidata_query')
@retry(wait_exponential_multiplier=500, stop_max_attempt_number=5, wait_exponential_max=15000)
def wikidata_query(query):
    try:
        sparql.setQuery(query)
        results = sparql.query().bindings
        return results
    except Exception as err:
        print(err)
        raise err


def wikidata_get_person(name):
    results = wikidata_query(f"""
            SELECT ?item ?itemLabel ?itemDescription ?sitelink (GROUP_CONCAT(DISTINCT ?item_alt_label; separator='; ') AS ?itemAltLabels) WHERE {{
            ?item (wdt:P279|wdt:P31) wd:Q5
             VALUES ?name {{
               "{name}"@en
             }}
             OPTIONAL {{
                ?item skos:altLabel ?item_alt_label .
                FILTER (lang(?item_alt_label)='en')
            }}
           ?sitelink schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> ; 
           schema:name ?name .
           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        GROUP BY ?item ?itemLabel ?itemDescription ?sitelink
        """)

    return results


def wikidata_get_organisation(name):
    results = wikidata_query(f"""
            SELECT ?item ?itemLabel ?itemDescription ?sitelink (GROUP_CONCAT(DISTINCT ?item_alt_label; separator='; ') AS ?itemAltLabels) WHERE {{
                ?item wdt:P31/wdt:P279* wd:Q43229
                 VALUES ?name {{
                   "{name}"@en
                 }}
                 OPTIONAL {{
                    ?item skos:altLabel ?item_alt_label .
                    FILTER (lang(?item_alt_label)='en')
                }}
               ?sitelink schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ; 
               schema:name ?name .
               SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
            }}
            GROUP BY ?item ?itemLabel ?itemDescription ?sitelink
               """)

    return results


def augment_entity_name(name):
    names = []
    try:
        page = get_wikipedia_page(name)
        results = wikidata_get_person(page.title)
        if len(results) == 0:
            is_person = False
            results = wikidata_get_organisation(page.title)
            if len(results) == 0:
                return names
        else:
            is_person = True

        names.append(name)
        names.append(page.title)

        if is_person:
            m = re.search('(.*?)(?:,| \(| is | was )', page.summary)
            if m:
                wikipedia_full_name = m.group(1).strip()
                names.append(wikipedia_full_name)

        for result in results:
            names.append(result["itemLabel"].value)
            if result.get('itemAltLabels'):
                names += result.get("itemAltLabels").value.split('; ')

        # remove duplicates, while keeping order
        names = unique(names)
        return names

    except Exception as err:
        # error on wikipedia disambiguation page
        print(err)
        return names


def normalize_names(name):
    labels = [name]
    results = wikipedia.search(name, results=1)
    if len(results) == 0:
        return labels

    return augment_entity_name(results[0])


def normalize_dataset(dataset, env):
    with jsonlines.open(f'../data/annotated_datasets/{dataset}.{env}.normalized.jsonl',
                        mode='w') as writer, jsonlines.open(f'../data/annotated_datasets/{dataset}.{env}.jsonl',
                                                            mode='r') as reader:
        for i, qa in enumerate(tqdm(reader)):
            if qa['question'].startswith("who "):
                print(qa['question'])
                print(', '.join(qa['answer']))
                qa = normalize_qa(qa)
                print(', '.join(qa['answer']))
                print('')
            writer.write(qa)


if __name__ == '__main__':
    dataset = args.dataset
    env = args.env
    normalize_dataset(dataset, env)
