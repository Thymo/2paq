import re

import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def get_clusters(data, threshold=0.7):
    """
    Finds clusters of similar data points.

    Clustering is done using fast community clustering algorithm.

    The items are compared based on a dense question embedding.
    """

    embeddings = [d['embedding'] for d in data]
    clusters_result = util.community_detection(embeddings, min_community_size=1, threshold=threshold)
    clusters = []
    for i, cluster_ids in enumerate(clusters_result):
        cluster = [data[idx] for idx in cluster_ids]
        clusters.append([d for d in cluster])

    return clusters


def embedd_questions(data, key='question'):
    """
    Embed the sentences/text using the MiniLM language model (which uses mean pooling)
    """
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    print('Embedding questions')
    sentences = [i[key] for i in data]
    embeddings = encoder.encode(sentences)
    data = [dict(item, embedding=embedding.tolist()) for item, embedding in zip(data, embeddings)]
    return data

def mask_ner(data, key='question', masked_key='masked', masked_token="<MASK>"):
    print('Mask named entities')
    spacy_nlp = spacy.load('en_core_web_trf')

    model_name = "elastic/distilbert-base-uncased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    def mask_qa(qa):
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True, )
        ner_results = nlp(qa['question'])

        question = qa[key]
        q = list(question)
        for ner in ner_results:
            q[ner['start']:ner['end']] = '_' * (ner['end'] - ner['start'])
        masked = ''.join(q)

        doc = spacy_nlp(masked)
        for ent in doc.ents:
            if ent.label_ == "ORDINAL": continue  # keep first, second, etc.
            masked = masked[:ent.start_char] + '_' * len(ent.text) + masked[ent.end_char:]

        masked = re.sub('(_+ {0,1}_+)+', masked_token, masked)
        masked = re.sub('_', masked_token, masked)
        qa[masked_key] = masked

    return [mask_qa(qa) for qa in data]