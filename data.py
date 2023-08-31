import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class MovieLens(Dataset):
    """
    A PyTorch Dataset class for the MovieLens dataset.

    Args:
        file_path (str): The path to the MovieLens ratings.csv file.
    """

    def __init__(self, file_path: str = 'data/ratings.csv'):
        """
        Load the MovieLens dataset and create a mapping from user and movie IDs to
        their normalized indices.
        """
        self.data = pd.read_csv(file_path)

        self.users = np.unique(self.data['userId'])
        self.movies = np.unique(self.data['movieId'])

        self.user_map = {id: np.where(self.users == id)[0][0]
                         for id in self.users}
        self.movie_map = {id: np.where(self.movies == id)[0][0]
                          for id in self.movies}

        self.data['normalized_user_id'] = self.data['userId'].map(
            self.user_map.get)
        self.data['normalized_movie_id'] = self.data['movieId'].map(
            self.movie_map.get)

        self.size: tuple[int, int] = len(self.users), len(self.movies)

    def __len__(self) -> int:
        """
        Get the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Retrieve a data point from the dataset.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple of (user_id, movie_id, rating).
        """
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
