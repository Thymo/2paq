import json

from jsonlines import jsonlines


def load_json(path):
    with open(path) as file:
        return json.load(file)


def dump_json(items, fi):
    with open(fi, 'w') as f:
        f.write(json.dumps(items))


def dump_jsonl(items, fi):
    with open(fi, 'w') as f:
        for k, item in enumerate(items):
            f.write(json.dumps(item) + '\n')


def load_jsonl(fi):
    out = []
    with jsonlines.open(fi, mode='r') as reader:
        for k, item in enumerate(reader):
            out.append(item)
    return out
