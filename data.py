import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class MovieLens(Dataset):
    def __init__(self, file_path='data/ratings.csv'):
        self.data = pd.read_csv(file_path)

        self.users = np.unique(self.data['userId'])
        self.movies = np.unique(self.data['movieId'])

        self.user_map = {id: np.where(self.users == id)[
            0][0] for id in self.users}
        self.movie_map = {id: np.where(self.movies == id)[
            0][0] for id in self.movies}

        self.data['normalized_user_id'] = self.data['userId'].map(
            self.user_map.get)
        self.data['normalized_movie_id'] = self.data['movieId'].map(
            self.movie_map.get)

        self.size = len(self.users), len(self.movies)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id = torch.LongTensor(
            [self.data.iloc[index]['normalized_user_id']])
        movie_id = torch.LongTensor(
            [self.data.iloc[index]['normalized_movie_id']])
        rating = torch.FloatTensor(
            [self.data.iloc[index]['rating']])

        return user_id, movie_id, rating


if __name__ == '__main__':
    data = MovieLens()
    for i in range(10):
        print(data[i])
