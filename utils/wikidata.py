from SPARQLWrapper import SPARQLWrapper2
from retrying import retry
from utils.evaluation_utils import normalize_date

from utils.cache import cache

sparql = SPARQLWrapper2("https://query.wikidata.org/sparql")

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


def augment_entity(name):
    try:
        query = f"""
                   SELECT ?wdLabel ?ps_Label ?wdpqLabel ?pq_Label ?item WHERE {{
                       VALUES ?name {{
                           "{name}"@en
                       }}

                       ?item ?p ?statement .
                       ?statement ?ps ?ps_ .
                       ?wd wikibase:claim ?p.
                       ?wd wikibase:statementProperty ?ps.

                       OPTIONAL {{
                           ?statement ?pq ?pq_ .
                           ?wdpq wikibase:qualifier ?pq .
                       }}

                       ?sitelink schema:about ?item ;
                       schema:isPartOf <https://en.wikipedia.org/> ; 
                       schema:name ?name .             
                       SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}

                   }} ORDER BY ?p"""
        results = wikidata_query(query)
        if len(results) == 0:
            return None, None

        item_id = results[0]['item'].value

        results = [(r['wdLabel'].value,
                    normalize_date(r['ps_Label'].value),
                    r.get('wdpqLabel').value if r.get('wdpqLabel') else '',
                    normalize_date(r.get('pq_Label').value) if r.get('pq_Label') else ''
                    )
                   for r in results]

        return results, item_id

    except Exception as err:
        print(err)
        return None, None