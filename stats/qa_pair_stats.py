import glob
import subprocess
from utils.cache import cache


@cache('get_qa_pairs_count')
def get_qa_pairs_count(relation):
    path = f'../data/2PAQ/{relation}/{relation}.jsonl'
    result = subprocess.check_output(['wc', '-l', path])
    count = float(str(result).replace("b'", '').split(' ')[0])
    return count


def get_relations():
    relations = []
    paths = glob.glob("../data/2PAQ/*")
    for path in paths:
        relation = path.split('/')[-1]
        relations.append(relation)
    return relations


if __name__ == '__main__':
    relations = get_relations()
    for r in relations:
        count = get_qa_pairs_count(r)
        print(r, '\t', count)
