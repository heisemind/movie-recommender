import pandas as pd
from torch.nn.functional import pairwise_distance
from model import MatrixFactorization
from data import MovieLens
from utils import load_model


class SimilarMovies:
    def __init__(self):
        movielens_dataset = MovieLens()
        num_users, num_movies = movielens_dataset.size
        embedding_size = 200
        self.movielens = movielens_dataset
        self.model = MatrixFactorization(num_users,
                                         num_movies,
                                         embedding_size)

        load_model(self.model, 'recommender_model.pth')
        self.movies = pd.read_csv('data/movies.csv')

    def get_similars(self, target_movie_id, top_n=5):
        target_movie_id = self.movielens.movie_map[target_movie_id]

        movie_embeddings = self.model.movie_embedding.weight.data
        target_embedding = movie_embeddings[target_movie_id].reshape(1, -1)

        similarities = pairwise_distance(
            target_embedding.unsqueeze(0), movie_embeddings)

        similar_movie_indices = similarities.argsort(
            dim=1, descending=False).squeeze()[1:top_n+1]

        similar_movie_indices = [id.item() for id in similar_movie_indices]

        self.display_similar_movies(target_movie_id, similar_movie_indices)

    def movie_info(self, movie_id):
        movie_id = self.movielens.movies[movie_id]
        return {
            'Movie ID': [movie_id],
            'Title': self.movies[self.movies['movieId'] == movie_id]['title'].values[0],
            'Genre': self.movies[self.movies['movieId'] == movie_id]['genres'].values[0]}

    def display_similar_movies(self, movie_id, similar_ids):
        main_title = self.movie_info(movie_id)

        print(
            f'Top {len(similar_ids)} most similar movies to "{main_title["Title"]}" [{main_title["Genre"]}]')

        for id in similar_ids:
            title = self.movie_info(id)
            print(
                f'- [{title["Movie ID"][0]}] {title["Title"]} [{title["Genre"]}]')


if __name__ == '__main__':
    finder = SimilarMovies()
    ids = [
        116797,  # The Imitation Game (2014)
        7153,  # Lord of the Rings: The Return of the King, The (2003)
        2959,  # Fight Club (1999)
        6377,  # Finding Nemo (2003)
        858,  # Godfather, The (1972)
        5349,  # Spider-Man (2002)
        109487  # Interstellar (2014)
    ]

    for target_id_movie in ids:
        finder.get_similars(target_id_movie)
