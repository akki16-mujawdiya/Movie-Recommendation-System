import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select important columns
movies = movies[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew'
    ]
]

# Remove missing values
movies.dropna(inplace=True)

# -------------------------
# Helper Functions
# -------------------------

# Extract names from genres and keywords
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Extract top 3 cast members
def convert3(text):
    L = []
    counter = 0

    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break

    return L

# Extract director name
def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break

    return L

# -------------------------
# Apply Functions
# -------------------------

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert3)

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview string to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from names
movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

# -------------------------
# Create Tags Column
# -------------------------

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

# New dataframe
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(
    lambda x: " ".join(x)
)

# -------------------------
# Vectorization
# -------------------------

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    new_df['tags']
).toarray()

# -------------------------
# Cosine Similarity
# -------------------------

similarity = cosine_similarity(vectors)

# -------------------------
# Recommendation Function
# -------------------------

def recommend(movie):

    movie_index = new_df[
        new_df['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# -------------------------
# User Input
# -------------------------

movie_name = input(
    "Enter Movie Name: "
)

recommend(movie_name)