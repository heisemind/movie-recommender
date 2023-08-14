import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

    def forward(self, user_idx, movie_idx):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        rating = torch.sum(user_emb * movie_emb, dim=1)
        return rating
