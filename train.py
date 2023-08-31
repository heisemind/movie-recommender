from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch
from model import MatrixFactorization
from utils import save_model
from data import MovieLens

# Training configuration
BATCH_SIZE = 4096
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_SIZE = 200
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30

# Load the MovieLens dataset
movielens = MovieLens()

# Split dataset into train and validation sets
val_ratio = 0.2
train_ratio = 1.0 - val_ratio

total_length = len(movielens)
train_length = int(train_ratio * total_length)
val_length = int(val_ratio * total_length)

train_dataset, val_dataset = random_split(
    movielens, [train_length, val_length],
    generator=torch.Generator().manual_seed(23))

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Create the MatrixFactorization model
num_users, num_movies = movielens.size
model = MatrixFactorization(num_users, num_movies, EMBEDDING_SIZE).to(DEVICE)

# Loss and optimizer
criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
min_val_loss = np.inf

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}')
    running_loss = 0.0

    # Train Loop
    model.train()
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (users_id, movies_id, ratings) in train_loop:
        users_id = users_id.squeeze().to(DEVICE)
        movies_id = movies_id.squeeze().to(DEVICE)
        ratings = ratings.squeeze().to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(users_id, movies_id)

        # Compute the loss
        loss = criterion(outputs, ratings)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loop.set_postfix(loss=running_loss / (i + 1))

    running_loss = running_loss / len(train_loader)
    print(f'Train Loss: {running_loss}')

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for users_id, movies_id, ratings in val_loader:
            users_id = users_id.squeeze().to(DEVICE)
            movies_id = movies_id.squeeze().to(DEVICE)
            ratings = ratings.squeeze().to(DEVICE)

            # Forward pass
            outputs = model(users_id, movies_id)

            # Compute the loss
            loss = criterion(outputs, ratings)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {val_loss}')

    # Save the model if validation loss decreases
    if val_loss < min_val_loss:
        save_model(model, 'recommender_model.pth')
        min_val_loss = val_loss

print('Finished Training')
