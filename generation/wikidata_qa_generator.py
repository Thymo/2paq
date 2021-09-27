import re
import sqlite3
import unicodedata
import os
from pathlib import Path

from jsonlines import jsonlines
from tqdm import tqdm

from utils.cache import cache
from utils.evaluation_utils import normalize_date

templates = {
    'performer': 'who sings <SUBJECT>',
    'inception': 'when was <SUBJECT> created',
    # 'inception': 'when did <SUBJECT> come into existence',
    'publication_date': 'when was <SUBJECT> released',
    # 'publication_date': 'when did <SUBJECT> come out',
    'lyrics_by': 'who wrote the song <SUBJECT>',
    'composer': 'who wrote <SUBJECT>',
    'start_time': 'when did <SUBJECT> start',
    'country': 'in which country is <SUBJECT>',
    'point_in_time': 'when was <SUBJECT>',
    'end_time': 'when did <SUBJECT> end',
    'located_in_the_administrative_territorial_entity': 'where is <SUBJECT>',
    'number_of_seasons': 'how many seasons of <SUBJECT> are there',
    'dissolved_date': 'when did <SUBJECT> stop to exist',
    'author': 'who wrote <SUBJECT>',
    'number_of_episodes': 'how many episodes of <SUBJECT> are there',
    'winner': 'who won <SUBJECT>',
    'sport': 'what sport does <SUBJECT> do',
    'country_of_citizenship': 'in which country is <SUBJECT> born',
    'location': 'where is <SUBJECT>',
    'spouse': 'who is the spouse of <SUBJECT>',
    'occupation': 'what does <SUBJECT> do',
    'notable_work': 'what is <SUBJECT> known for',
}

relation_ids = {
    'performer': 'P175',
    'inception': 'P571',
    'publication_date': 'P577',
    'lyrics_by': 'P676',
    'composer': 'P86',
    'start_time': 'P580',
    'country': 'P17',
    'point_in_time': 'P585',
    'named_after': 'P138',
    'end_time': 'P582',
    'located_in_the_administrative_territorial_entity': 'P131',
    'number_of_seasons': 'P2437',
    'dissolved_date': 'P576',
    'author': 'P50',
    'number_of_episodes':'P1113',
    'winner': 'P1346',

    # triviaqa extra
    'sport': 'P641',
    'country_of_citizenship': 'P27',
    'location': 'P276',
    'spouse': 'P26',
    'occupation': 'P106',
    'notable_work': 'P800',
}

connection = sqlite3.connect("../data/wikidata/wikidata-labels")

@cache('get_id_name_mapping')
def get_id_name_mapping(ids):
    cursor = connection.cursor()
    mapping = cursor.execute("SELECT key, value FROM labels WHERE key IN (%s)" % ','.join('?' * len(ids)),
                             ids).fetchall()
    mapping = {m[0]: m[1] for m in mapping}
    return mapping


def normalize(text):
    text = text.encode().decode('unicode_escape')
    return unicodedata.normalize("NFD", text)


def generate_qa_pairs(relation):
    p = re.compile(
        '<http://www.wikidata.org/entity/(.+)> <http://www.wikidata.org/prop/direct/(.+)> <http://www.wikidata.org/entity/(.+)> .')
    p_date = re.compile(
        '<http://www.wikidata.org/entity/(.+)> <http://www.wikidata.org/prop/direct/(.+)> "(.+)"\^\^.+ .')
    with jsonlines.open(f'../data/2PAQ/{relation}/{relation}.jsonl', mode='w') as writer:
        with open(f"../data/2PAQ/{relation}/{relation}.nt", "r") as f:
            triplets = []
            count = 0

            def write_triplets(triplets):
                subjects_ids = [t[0] for t in triplets]
                objects_ids = [t[2] for t in triplets]
                ids = subjects_ids + objects_ids
                id_name_mapping = get_id_name_mapping(tuple(ids))
                print('id_name_mapping')
                for triplet in triplets:
                    subj_, pred, obj_ = triplet
                    try:
                        subj = id_name_mapping[subj_]
                        if obj_.startswith('Q'):
                            obj = id_name_mapping[obj_]
                        elif obj_.startswith('+') or obj_.startswith('-'):
                            obj = obj_.replace('+','').replace('-','')
                        else:
                            obj = normalize_date(obj_)
                    except KeyError:
                        continue
                    template = templates[relation]
                    question = template.replace('<SUBJECT>', subj).lower()
                    question = normalize(question)
                    answer = obj
                    answer = normalize(answer)
                    qa = {'question': question, 'answer': [answer]}
                    writer.write(qa)

            for line in tqdm(f):
                count += 1
                line = str(line).strip()
                m = p.search(line)
                if not m:
                    m = p_date.search(line)
                if not m: continue

                triplet = m.groups()
                triplets.append(triplet)

                if count % 10000 == 0:
                    write_triplets(triplets)
                    triplets = []
            write_triplets(triplets)

def extract_triplets(relation):
    pred = relation_ids[relation]
    os.environ['PATH'] += os.pathsep + '/home/testing/drive/thymo/.local/bin' # load local packages for rg (ripgrep)
    os.system(f'time rg "<http://www.wikidata.org/prop/direct/{pred}> " -z ../data/wikidata/subset.nt.gz > ../data/2PAQ/{relation}/{relation}.nt')

if __name__ == '__main__':

    relations = templates.keys()

    for relation in relations:
        print(relation)
        Path(f'../data/2PAQ/{relation}').mkdir(parents=True, exist_ok=True)

        extract_triplets(relation)
        generate_qa_pairs(relation)
