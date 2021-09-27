from jsonlines import jsonlines
from clustering_utils import embedd_questions, get_clusters, mask_ner

def get_question_clusters(path):
    with jsonlines.open(path) as reader:
        data = [qa['input_qa'] for qa in reader]

    embedded_question_data = embedd_questions(data, 'question')

    clusters = get_clusters(embedded_question_data, 0.5)
    len(clusters)
    for cluster in clusters:
        for qa in cluster:
            print(qa['question'])
        print('\n')
    return clusters


def get_masked_question_clusters(path):
    with jsonlines.open(path) as reader:
        data = [qa['input_qa'] for qa in reader]

    masked_data = mask_ner(data)
    embedded_data = embedd_questions(masked_data, 'masked')

    clusters = get_clusters(embedded_data, 0.9)
    len(clusters)
    for cluster in clusters:
        for qa in cluster:
            print(qa['masked'])
            # print(qa['question'])
        print('\n')
    return clusters

if __name__ == '__main__':
    path = '../data/results/triviaqa-dev/baseline_hnsw/incorrect.top5.jsonl'
    get_question_clusters(path)
    get_masked_question_clusters(path)