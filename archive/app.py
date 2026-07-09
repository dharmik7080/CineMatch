import streamlit as st
import pickle 
import pandas as pd
import requests
import gzip

#movies_list = pickle.load(open('movies.pkl','rb'))
#movies_list = movies_list['title'].values

movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

def recommend (movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies = []
    recommended_movies_poster = []

    for i in movies_list:
       movie_id = movies.iloc[i[0]].movie_id

       # fetch posters from API 
       recommended_movies.append(movies.iloc[i[0]].title)
       recommended_movies_poster.append(get_movie_details(movie_id))

    return recommended_movies,recommended_movies_poster   



similarity = pickle.load(open('similarity.pkl','rb'))

 


#with gzip.open('similarity.pkl.gz', 'rb') as f:
 #   similarity = pickle.load(f)



def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MWZjNzRjZTU2MDI4ODI3ODZlMWU5ZDQ5MzNmZGNjNiIsIm5iZiI6MTcyMjkzOTI3OC4xNzY5NjUsInN1YiI6IjY2YjFmNTNjOGUxOGRjNTA1YTllNjhjNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.UYyGr5Cw__O1k0SK-Blyreukq4qLyS7e4nwDhPjxk5A"
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data['poster_path']





st.title('Movie Recommmender System')
selected_movie_name = st.selectbox(
    'Enter The Movie Name',
    movies['title'].values)

if st.button('Recommend'):
    names,poster = recommend(selected_movie_name)
    
    # Use columns to display the posters
    cols = st.columns(5)  # Adjust the number of columns as needed
    for i, p in enumerate(poster):
        if i < len(cols):
            with cols[i]:
                st.image(p)
                st.write(names[i])







# Example usage
# Example usage
#movie_id = 69  # Replace with the actual movie ID
#movie_details = get_movie_details(movie_id)
#print(movie_details)