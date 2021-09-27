import glob
import os
import re

home = '/cluster/project6/mr_corpora/________'


def get_embed_command(relation):
    return 'python -m paq.retrievers.embed ' \
           '--model_name_or_path ./data/models/retrievers/retriever_multi_base_256 ' \
           '--batch_size 128 ' \
           f'--qas_to_embed ../data/2PAQ/{relation}/{relation}.jsonl ' \
           f'--output_dir ../data/2PAQ/{relation}/vectors ' \
           '--n_jobs -1 ' \
           '--fp16 --verbose --memory_friendly_parsing'


def get_to_embed_relations():
    relations = []
    paths = glob.glob("../data/2PAQ/*")

    for path in paths:
        if not os.path.isdir(f'{path}/vectors'):
            relations.append(path.split('/')[-1])
    return relations


def generate_embed_job(relations):
    command_lines = [get_embed_command(r) for r in relations]

    header = f"""
#$ -cwd
#$ -S /bin/bash
#$ -o {home}/data/logs/embed.out
#$ -e {home}/data/logs/embed.err
#$ -t 1-{len(command_lines)}
#$ -l tmem=16G
#$ -l h_rt=4:00:00
#$ -l gpu=true
#$ -N paq-embed

conda activate paq
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd {home}/paq

"""

    body_lines = []

    for job_id, command_line in enumerate(command_lines, 1):
        body_lines.append(f'test $SGE_TASK_ID -eq {job_id} && sleep 10 && {command_line}')

    file = header + "\n".join(body_lines)
    return file


def get_retrieval_command(relation):
    return 'python -m paq.retrievers.retrieve_efficient ' \
           '--model_name_or_path ./data/models/retrievers/retriever_multi_base_256 ' \
           '--qas_to_answer ../data/annotated_datasets/datasets.augmented.jsonl ' \
           '--qas_to_retrieve_from ./data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl ' \
           '--top_k 50 ' \
           '--fp16 ' \
           '--memory_friendly_parsing ' \
           '--verbose ' \
           '--faiss_index_path ./data/indices/multi_base_256_hnsw_sq8.faiss ' \
           f'--output_file ../data/results/{relation}.jsonl ' \
           '--extension_dir ../data/2PAQ ' \
           f'--extensions {relation}'


def generate_retrieval_job(relations):
    command_lines = [get_retrieval_command(r) for r in relations]

    header = f"""
#$ -cwd
#$ -S /bin/bash
#$ -o {home}/data/logs/retrieve.out
#$ -e {home}/data/logs/retrieve.err
#$ -t 1-{len(command_lines)}
#$ -l tmem=40G
#$ -l h_rt=4:00:00
#$ -l gpu=true
#$ -N paq-retrieve

conda activate paq
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd {home}/paq

"""

    body_lines = []

    for job_id, command_line in enumerate(command_lines, 1):
        body_lines.append(f'test $SGE_TASK_ID -eq {job_id} && sleep 10 && {command_line}')

    file = header + "\n".join(body_lines)
    return file


def get_to_retrieve_relations():
    relations = []
    paths = glob.glob("../data/2PAQ/*")

    for path in paths:
        if os.path.isdir(f'{path}/vectors'):
            relation = path.split('/')[-1]
            if not os.path.isfile(f'../data/results/{relation}.jsonl'):
                relations.append(relation)
    return relations


def get_rerank_command(relation):
    return 'python -m paq.rerankers.rerank ' \
           '--model_name_or_path data/models/rerankers/reranker_multi_xxlarge ' \
           f'--qas_to_rerank ../data/results/{relation}.jsonl ' \
           f'--output_file ../data/results/{relation}.reranked.jsonl ' \
           '--top_k 50 ' \
           '--fp16 ' \
           '--batch_size 4 --verbose --n_jobs -1'


def generate_rerank_job(relations):
    command_lines = [get_rerank_command(r) for r in relations]

    header = f"""
#$ -cwd
#$ -S /bin/bash
#$ -o {home}/data/logs/rerank.out
#$ -e {home}/data/logs/rerank.err
#$ -t 1-{len(command_lines)}
#$ -l tmem=30G
#$ -l h_rt=4:00:00
#$ -l gpu=true
#$ -N paq-rerank

conda activate paq
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd {home}/paq

"""

    body_lines = []

    for job_id, command_line in enumerate(command_lines, 1):
        body_lines.append(f'test $SGE_TASK_ID -eq {job_id} && sleep 10 && {command_line}')

    file = header + "\n".join(body_lines)
    return file


def get_to_rerank_relations():
    relations = []
    paths = glob.glob("../data/results/*.jsonl")

    for path in paths:
        if '.reranked.jsonl' in path:
            continue
        path = path.replace(".jsonl", "")
        relation = path.split('/')[-1]
        if not os.path.isfile(f'../data/results/{relation}.reranked.jsonl'):
            relations.append(relation)
    return relations

def filter(relations):
    return [r for r in relations if not re.match(r"(author|country|publication_date|country_of_citizenship|occupation|located_in_the_administrative_territorial_entity)", r)]

relations_to_embed = filter(get_to_embed_relations())
with open("embed.job.sh", "w") as file:
    file.write(generate_embed_job(relations_to_embed))

relations_to_retrieve = filter(get_to_retrieve_relations())
with open("retrieve.job.sh", "w") as file:
    file.write(generate_retrieval_job(relations_to_retrieve))

relations_to_rerank = get_to_rerank_relations()
with open("rerank.job.sh", "w") as file:
    file.write(generate_rerank_job(relations_to_rerank))
