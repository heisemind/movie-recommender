import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """
    A PyTorch embeddings for matrix factorization.

    Args:
        num_users (int): The number of users in the dataset.
        num_movies (int): The number of movies in the dataset.
        embedding_size (int): The size of the embedding vectors.
    """

    def __init__(self, num_users: int, num_movies: int, embedding_size: int):
        """
        Create the user and movie embedding layers.
        """
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(
            num_users, embedding_size, padding_idx=0)  # type: torch.nn.Embedding
        self.movie_embedding = nn.Embedding(
            num_movies, embedding_size, padding_idx=0)  # type: torch.nn.Embedding

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute the predicted rating for a given user and movie.

        Args:
            user_idx (torch.Tensor): The index of the user.
            movie_idx (torch.Tensor): The index of the movie.

        Returns:
            torch.Tensor: The predicted rating.
        """
        user_emb = self.user_embedding(user_idx)  # type: torch.Tensor
        movie_emb = self.movie_embedding(movie_idx)  # type: torch.Tensor
        rating = torch.sum(user_emb * movie_emb, dim=1)  # type: torch.Tensor
        return rating
