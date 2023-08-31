# Movie Recommender System

This is a simple movie recommender system built using matrix factorization. The system leverages the MovieLens dataset to predict user movie ratings and provides recommendations based on similar movies.

**Table of Contents**

- Description
- Installation
- Usage
- License
- Check Out Heise Mind

**Description**

This project contains three main components:

1. `data`: Contains the dataset handling, loading, and preprocessing code.
2. `model`: Includes the implementation of the matrix factorization model.
3. `utils`: Contains utility functions for saving and loading models.

The recommender system goes through training epochs using the training data, predicts ratings using the trained model, and evaluates its performance using validation data.

**Installation**

1. Clone the repository:

```bash
   git clone https://github.com/eduheise/movie-recommender.git
   cd movie-recommender
```

2. Install the required dependencies:

```bash
   pip install -r requirements.txt
```

3. Download and place the [MovieLens dataset](https://grouplens.org/datasets/movielens/) (`ratings.csv`, `movies.csv`, etc.) in the `data` directory.

**Usage**

1. Configure the project by adjusting parameters in the scripts, such as embedding size, learning rate, and batch size.

2. Train the recommender system:

```bash
   python train.py
```

   This will train the model using matrix factorization and save the best model based on validation loss.

3. Use the trained model to get movie recommendations:

```bash
   python similar_movies.py
```

   This script demonstrates the process of recommending similar movies for a given list of movie IDs.

**License**

This project is licensed under the MIT License.

---

**Check Out Heise Mind**

If you're interested in AI, check out my YouTube channel, [Heise Mind](https://www.youtube.com/@HeiseMind). I create deep-tech content about a variety of tech-related topics.

You might find my video on "Matrix Factorization Recommender System using PyTorch" particularly helpful: [Watch the Video](https://www.youtube.com/watch?v=4PusFiTkytE).
