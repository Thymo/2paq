from pathlib import Path

from SPARQLWrapper import SPARQLWrapper2
from jsonlines import jsonlines
from retrying import retry
from tqdm import tqdm

from utils.cache import cache
from utils.evaluation_utils import normalize_date
from wikidata_qa_generator import normalize

sparql_dbpedia = SPARQLWrapper2("https://dbpedia.org/sparql")

templates = {
    'dbo:artist': 'who sings <SUBJECT>',
    'dbp:artist': 'who sings <SUBJECT>',
    'dbo:writer': 'who wrote <SUBJECT>',
    'dbo:releaseDate': 'when did <SUBJECT> come out',
    'dbo:author': 'who is the author of <SUBJECT>',
    'dbp:firstAired': 'when did <SUBJECT> first air',
    'dbp:numSeasons': 'how many seasons of <SUBJECT> are there',
    'dbp:numEpisodes': 'how many episodes of <SUBJECT> are there',
    'dbo:numberOfEpisodes': 'how many episodes of <SUBJECT> are there',
    'dbo:country': 'in which country is <SUBJECT>',
    'dbp:founded': 'when was <SUBJECT> founded',
    'dbo:formationDate': 'when was <SUBJECT> created',
}


@cache('dbpedia_query_generator')
@retry(wait_exponential_multiplier=500, stop_max_attempt_number=5, wait_exponential_max=15000)
def dbpedia_query(query):
    try:
        sparql_dbpedia.setQuery(query)
        results = sparql_dbpedia.query().bindings
        return results
    except Exception as err:
        print(err)
        raise err


def generate(predicate):
    total = int(dbpedia_query(f"""
        select (count(?s) as ?count) where {{
            ?s {predicate} ?o
        }}
        """)[0]['count'].value)
    print('Total items',total)
    path_name = predicate.replace(':', '_')
    Path(f'../data/2PAQ/{path_name}').mkdir(parents=True, exist_ok=True)
    with jsonlines.open(f'../data/2PAQ/{path_name}/{path_name}.jsonl', mode='w') as writer:
        for offset in tqdm(range(0, total, 10000)):
            results = dbpedia_query(f"""
            select * where {{
                ?subject {predicate} ?object .
                optional {{
                    ?subject rdfs:label ?subjectLabel .
                    filter langMatches(lang(?subjectLabel), 'EN')
                }}
                optional {{
                  ?object rdfs:label ?objectLabel .
                  filter langMatches(lang(?objectLabel), 'EN')
              }}
            }}
            
            LIMIT 10000
            OFFSET {offset}
            """)

            for r in results:
                pred = predicate
                print(r)

                if not r.get('subjectLabel'):
                    continue
                subj = r['subjectLabel'].value
                obj = normalize_date(r.get('objectLabel').value if r.get('objectLabel') else r['object'].value)
                template = templates[pred]
                question = template.replace('<SUBJECT>', subj).lower()
                question = normalize(question)
                answer = obj
                answer = normalize(answer)
                qa = {'question': question, 'answer': [answer]}
                writer.write(qa)
                print(qa)


if __name__=='__main__':
    generate("dbp:artist")
    generate('dbo:artist')  # dbo:artist = performer, dbo is higher quality but lower coverage than dbp
    generate('dbo:writer')
    generate('dbo:releaseDate')
    generate('dbp:firstAired')
    generate('dbo:author')
    generate('dbp:numSeasons')
    generate('dbp:numEpisodes')
    generate('dbo:numberOfEpisodes')
    generate('dbo:country')
    generate('dbp:founded')
    generate('dbo:formationDate')


