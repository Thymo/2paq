from jsonlines import jsonlines
from tqdm import tqdm
import re

dataset = 'nq-open'
env = 'dev'

path = f'../data/annotated_datasets/{dataset}.{env}'
in_path = f'{path}.augmented.jsonl'
out_path = f'{path}.movies.txt'

def clean_movie_entity(entity):
    return re.sub("\(.*\)", "", entity).strip()


def clean_movie_text(text):
    text = re.sub("((tv|netflix|[0-9]{4}) (show|series|movie|film)) ", "", text)
    text = re.sub("((the )?(netflix )?((tv )?(show|series)|movie|film))$ ", "", text)
    text = re.sub("(season|episode) [0-9]+","", text)
    text = text.strip()
    return text

def extract_movie(qa):
    m = re.search('who (?:play|starred)(?:.*?) (?:in|on)(?: the)?(?: (?:movies?|show|film|series|original))? (.+)',
                  qa['question'])
    if m:
        movie = m.group(1).strip()
        if qa.get('wikidata_entity') and qa.get('wikidata_relations').startswith('cast member'):
            return clean_movie_entity(qa['wikidata_entity'])
        return clean_movie_text(movie)

    return None


assert clean_movie_entity("King Kong (1976 film)") == "King Kong"
assert extract_movie({'question': "who plays spectre in from russia with love"}) == "from russia with love"

movies = []
with jsonlines.open(in_path, mode='r') as reader:
    for i, qa in enumerate(tqdm(reader)):
        movie = extract_movie(qa)
        if movie is not None:
            movies.append(movie)

movies = sorted(set(movies))
with open(out_path, mode='w') as out_file:
    for movie in movies:
        out_file.write(movie + '\n')