import streamlit as st
import pandas as pd
import h5py 
import joblib

dataset_path = 'dataset.csv'
anime = pd.read_csv(dataset_path)

anime = anime[['English', 'Japanese', 'Image Link', 'Type', 'Rating', 'Genres', 'Producers', 'Score', 'Rank', 'Popularity']]
anime['Tags'] = anime['Type'] + anime['Genres'] + anime['Rating']
data = anime.drop(columns=['Type', 'Genres', 'Rating'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(data['Tags'].values.astype('U')).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

def recommendation(anime):
    index = data[data['English'] == anime].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    recommendations = [data.iloc[i[0]].English for i in distance[0:5]]
    return recommendations

st.title('Anime Recommendation System')
user_input = st.text_input('Enter an anime title:')

if user_input:
    recommendations = recommendation(user_input)

    if recommendations:
        st.subheader('Top 5 Recommendations:')
        for index, recommendation in enumerate(recommendations[:5]):
            st.write(f"**Rank {index+1}**")
            st.write(f"**English:** {data.loc[recommendation, 'English']}")
            st.write(f"**Japanese:** {data.loc[recommendation, 'Japanese']}")
            st.image(data.loc[recommendation, 'Image Link'], caption='Image Link', use_column_width=True)
            st.write(f"**Producers:** {data.loc[recommendation, 'Producers']}")
            st.write(f"**Score:** {data.loc[recommendation, 'Score']}")
            st.write(f"**Rank:** {data.loc[recommendation, 'Rank']}")
            st.write(f"**Popularity:** {data.loc[recommendation, 'Popularity']}")
            st.write(f"**Tags:** {data.loc[recommendation, 'Tags']}")
            st.write("---")
    else:
        st.warning('No recommendations found')

st.subheader('Travel Through The Anime Universe')
st.dataframe(data.head())
