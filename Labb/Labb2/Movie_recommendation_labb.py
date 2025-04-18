import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dash import Dash, dcc, html, Input, Output
Ratings = pd.read_csv(r"C:\Code\ml-latest\ratings.csv")
Tags = pd.read_csv(r"C:\Code\ml-latest\tags.csv")
Movies = pd.read_csv(r"C:\Code\ml-latest\movies.csv")

movies = Movies.drop_duplicates("title")
movies_with_ratings = Ratings.merge(movies, on="movieId")
movies_with_ratings.drop(columns=["timestamp"], inplace=True)
movie_matrix = movies_with_ratings.merge(Tags, on=["userId", "movieId"])

movie_matrix = movie_matrix.groupby(
    ['userId', 'movieId', 'rating', 'title', 'genres'], as_index=False
).agg({'tag': lambda tags: '|'.join(set(tags.dropna()))})

# Filtrerar filmer
avg_ratings = movie_matrix.groupby("movieId")["rating"].mean()
high_rated_movies = avg_ratings[avg_ratings >= 3.5].index
filtered_movies = movie_matrix[movie_matrix["movieId"].isin(high_rated_movies)].copy()

# Lägg till features
filtered_movies["features"] = filtered_movies["tag"]

# Kombinerar features
movies_content = filtered_movies.groupby(["movieId", "title", "genres"], as_index=False).agg({
    "features": lambda x: ' '.join(set(x.dropna()))
})

#Delar upp taggarna för att hitta de mest liknande filmerna
vectorizer = CountVectorizer(tokenizer=lambda x: x.split("|"))
feature_matrix = vectorizer.fit_transform(movies_content["features"])
similarity = cosine_similarity(feature_matrix)

#Dash applikation
app = Dash(__name__)
server = app.server 
app.layout = html.Div([
    html.H1("Filmrekommendationer", style={"color": "white", "textAlign": "center"}),
    dcc.Dropdown(
        id='movie-dropdown',
        options=[{'label': title, 'value': title} for title in movies_content['title']],
        placeholder="Sök efter en film",
        searchable=True,
        persistence=True,
        style={
            "backgroundColor": "#1e1e1e",
            "color": "black",
            "border": "1px solid #555"
        }
    ),
    html.H2("Liknande filmer:", style={"color": "white", "marginTop": "30px"}),
    html.Ul(id='recommendation-list')
], style={
    "backgroundColor": "black",
    "padding": "30px",
    "minHeight": "100vh"
})

@app.callback(
    Output('recommendation-list', 'children'),
    Input('movie-dropdown', 'value')
)

def recommend(selected_title):
    if not selected_title:
        return []
    index = movies_content[movies_content['title'] == selected_title].index[0]
    similarity = list(enumerate(similarity[index]))
    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)[1:6] 

    recommendations = []

    for i in similarity:
        movie = movies_content.iloc[i[0]]
        recommendations.append(html.Li(
            html.Div([
                html.Div(f"{movie['title']}", style={
                    "fontWeight": "bold",
                    "fontSize": "18px",
                    "color": "#ffcc00"  
                }),
                html.Div(movie["genres"], style={
                    "fontSize": "14px",
                    "color": "#gray"
                })
            ]),
            style={
                "backgroundColor": "#gray",
                "padding": "15px",
                "borderRadius": "8px",
                "marginBottom": "10px",
                "color": "yellow",
            }
        ))
    return recommendations
if __name__ == '__main__':
    app.run_server(debug=True)