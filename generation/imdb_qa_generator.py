from imdb import IMDb
from jsonlines import jsonlines
from tqdm import tqdm

from wikidata_qa_generator import normalize

ia = IMDb()

relation = 'plays_character_in_dev'

movies = []
for movie_name in open('../data/imdb/nq-dev-movies.txt', 'r'):
    movies.append(movie_name.strip())

with jsonlines.open(f'../data/2paq/{relation}/{relation}.jsonl', mode='w') as writer:
    for movie_name in tqdm(movies):
        print('movie_name', movie_name)
        results = ia.search_movie(movie_name)
        filtered = [result for result in results if result['kind'] in ['tv series', 'movie', 'tv movie']]
        if len(filtered) == 0:
            filtered = [result for result in results if result['kind'] in ['short', 'video movie']]

        for result in filtered[:3]:
            print(result)
            try:
                movie = ia.get_movie(result.movieID)
            except:
                continue
            print('movie', movie)
            try:
                cast = movie['cast']
            except:
                continue
            for actor in cast:
                character = str(actor.currentRole).replace(' / ...','')
                question = f"who plays {character} in {movie['title']}"
                question = normalize(question).lower()
                answer = normalize(actor['name'])
                answer = actor['name']
                qa = {'question': question, 'answer': [answer]}
                print(qa)
                writer.write(qa)