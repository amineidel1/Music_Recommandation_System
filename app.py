import streamlit as st
import pandas as pd
import pickle
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# Fonction pour obtenir les recommandations
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Chargement des données

def load_data():
    df = pd.read_csv('song_dataset.csv', nrows=1000)
    # Ensure you have 'song_id' and 'title' columns in your CSV
    return df[['user','song', 'title', 'play_count']]


# Chargement du modèle

def load_model():
    with open('recommandation_model.pickle', 'rb') as file:
        loaded_dict = pickle.load(file)
    model = loaded_dict['predictions']  # L'objet SVD est stocké ici
    return model


# Interface Streamlit
def main():
    st.title('Music Recommendation App')
    df = load_data()
    song_id_to_title = pd.Series(df.title.values, index=df.song).to_dict()

    reader = Reader(rating_scale=(1, 10))  # Ajustez selon la gamme de vos données
    data = Dataset.load_from_df(df[['user', 'song', 'play_count']], reader)
    trainset = data.build_full_trainset()

    model = load_model()

    # Générer le testset et les prédictions
    testset = trainset.build_anti_testset()
    predictions = model.test(testset)
    # Obtenez les N meilleures recommandations
    top_n = get_top_n(predictions, n=5)

    # Sélection de l'utilisateur
    selected_user = st.selectbox("Select User:", list(top_n.keys()))

    if st.button('Get Recommendations'):
        user_ratings = top_n[selected_user]
        recommended_song_ids = [iid for (iid, _) in user_ratings]
        recommended_song_titles = [song_id_to_title[iid] for iid in recommended_song_ids]
        st.write("Recommended Songs:")
        for title in recommended_song_titles:
            st.write(title)

if __name__ == "__main__":
    main()