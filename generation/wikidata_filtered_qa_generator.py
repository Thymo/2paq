from pathlib import Path

from jsonlines import jsonlines
from tqdm import tqdm

from utils.utils import unique
from utils.wikidata import augment_entity
from wikidata_qa_generator import normalize, templates


def generate_qa_pairs(relation):
    template = templates[relation]
    Path(f'../data/2PAQ/{relation}_el').mkdir(parents=True, exist_ok=True)
    with jsonlines.open(f'../data/2PAQ/{relation}_el/{relation}_el.jsonl', mode='w') as writer, jsonlines.open(
            f"../data/annotated_datasets/datasets.augmented.jsonl") as reader:

        for qa in tqdm(reader):
            entity = qa.get('wikidata_entity')
            if entity is None: continue
            relation_text = relation.replace("_", " ")
            entity_data, entity_id = augment_entity(qa['wikidata_entity'])
            if entity_data is None: continue
            entity_data = unique([(predicate, obj) for predicate, obj, *_, in entity_data])

            for pred, obj, *_ in entity_data:
                if pred == relation_text:
                    question = template.replace('<SUBJECT>', entity).lower()
                    answer = obj
                    answer = normalize(answer)
                    qa = {'question': question, 'answer': [answer]}
                    writer.write(qa)
                    print(qa)


if __name__ == '__main__':
    relations = templates.keys()
    for relation in relations:
        print(relation)
        generate_qa_pairs(relation)
