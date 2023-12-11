from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# Load movie data at app startup
path = './ml-1m/'
ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
movies_cols = ['MovieID', 'Title', 'Genres']
users_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

ratings = pd.read_csv(path + 'ratings.dat', sep='::', engine='python', header=None, names=ratings_cols)
movies = pd.read_csv(path + 'movies.dat', sep='::', engine='python', header=None, names=movies_cols, encoding='iso-8859-1')
users = pd.read_csv(path + 'users.dat', sep='::', engine='python', header=None, names=users_cols)

R = pd.read_csv("ratings.csv")
precompute = {}

# Merge ratings and movies data
merged_data = pd.merge(ratings, movies, on='MovieID')

# Function to recommend top N popular movies in a given genre
def recommend_popular_movies(user_favorite_genre, N=5):
    genre_movies = merged_data[merged_data['Genres'].apply(lambda x: user_favorite_genre.lower() in x.lower().split('|'))]
    popularity_scores = genre_movies.groupby('MovieID')['Rating'].count().reset_index(name='Popularity')
    popular_movies = popularity_scores.sort_values(by='Popularity', ascending=False).head(N)
    popular_movies = pd.merge(popular_movies, movies, on='MovieID')
    return popular_movies[['MovieID', 'Title', 'Popularity']]

# Function to keep the top 30 similarities for each movie
def keep_top_similarities(similarity_matrix):
    print(similarity_matrix)
    top_similarities = similarity_matrix.apply(lambda row: row.nlargest(30).index, axis=1)
    top_similarities_matrix = similarity_matrix.apply(lambda row: row.where(row.index.isin(top_similarities.loc[row.name])), axis=1)
    return top_similarities_matrix

# Function for collaborative filtering
movie_similarity_matrix_top30 =  pd.read_csv("cosine_matrix.csv").iloc[:,1:]
def myIBCF(newuser):
    predictions = np.zeros_like(newuser, dtype=float)
    non_zero_values = newuser[newuser!=0]
    newuser[newuser!=0] = (non_zero_values - non_zero_values.min()) / (non_zero_values.max() - non_zero_values.min()) * 4 + 1
    denominator = (movie_similarity_matrix_top30.fillna(0) @ (newuser != 0))
    numerator = (movie_similarity_matrix_top30.fillna(0) @ (newuser))
    predictions = (numerator / denominator)
    predictions = predictions[newuser==0]
    predictions = predictions.sort_values(ascending=False)

    # Get the indices of the top 10 recommendations
    top_recommendations = predictions[:10]

    # Print the top 10 recommendations
    print("Top 10 Recommendations:")
    print(top_recommendations)

    # Get the indices of the top 10 recommendations
    # top_recommendations = np.argsort(predictions)[::-1][:10]

    # Return the top 10 recommendations
    return top_recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        genre = request.form.get('genre')
        if genre:
            try:
                if genre in precompute: recommendations = precompute[genre]
                else: 
                    recommendations = recommend_popular_movies(genre, N=10)
                    precompute[genre] = recommendations
                return render_template('index.html', genre=genre, recommendations=recommendations.to_dict(orient='records'))
            except Exception as e:
                return render_template('index.html', error=f'Error: {str(e)}')
        else:
            return render_template('index.html', error='Please enter a genre')
    return render_template('index.html')

@app.route('/rate_movies', methods=['GET', 'POST'])
def rate_movies():
    if request.method == 'POST':
        user_ratings = {}
        for movie_id in request.form:
            if movie_id.startswith('rating_'):
                rating = float(request.form[movie_id])
                user_ratings[movie_id.replace('rating_', '')] = rating

        new_user_ratings_hypothetical = np.zeros(R.shape[1])
        for movie_id, rating in user_ratings.items():
            new_user_ratings_hypothetical[R.columns == movie_id] = rating


        if np.any(new_user_ratings_hypothetical):
            # Call myIBCF function with user ratings
            recommendations = myIBCF(new_user_ratings_hypothetical)
            # recommended_movies = movies.loc[recommendations.index]
            # print(movies.loc[recommendations.index.map(lambda x: f'm{x}')])
            recommendations = movies.loc[recommendations.index]
            print(recommendations)
            return render_template('recommendations.html', recommendations=recommendations)
    else:
        random_movies = [
        {'MovieID': 'm260', 'Title': 'Star Wars: Episode IV - A New Hope (1977)'},
        {'MovieID': 'm2028', 'Title': 'Saving Private Ryan (1998)'},
        { 'MovieID': 'm1580', 'Title': 'Men in Black (1997)'},
         {'MovieID': 'm589', 'Title': 'Terminator 2: Judgment Day (1991)'},
         {'MovieID': 'm1198', 'Title': 'Raiders of the Lost Ark (1981)'}
        ]
        return render_template('rate_movies.html', movies=random_movies)
    
if __name__ == '__main__':
    app.run(debug=True)
