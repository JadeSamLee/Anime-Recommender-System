import streamlit as st
import h5py 
import joblib


def recommendation(anime):
    index = anime_list[anime_list['English'] == anime].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    recommendations = [anime_list.iloc[i[0]].English for i in distance[1:6]]
    return recommendations

anime_list=joblib.load(open('anime_recommend.h5','rb'))
anime_list=anime_list['English'].values

similarity=joblib.load(open('similarity.h5','rb'))

st.title("Anime Recommender System")
select_option=st.selectbox("Enter your favourite anime",anime_list)

if st.button("Recommend"):
    recommended=recommendation(select_option)
    for i in recommended:
       
        st.image(anime_list.loc[recommendation, 'Image Link'], caption='Image Link', use_column_width=True)
        st.write(i)
        st.write(f"**Japanese:** {anime_list.loc[recommendation, 'Japanese']}")
    