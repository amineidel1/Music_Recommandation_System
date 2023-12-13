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
    # Adding the explanatory section
    st.markdown("""
    ## About the App
    You want to know why we have recomanded this song for you 🤔

    To enhance your experience with our recommendation platform, we've incorporated a feature that not only suggests items you'll love but also explains why they're a good fit for you. Here's how it adds value to your experience: 

    This process is explained here, where the 'User-item Interaction Matrix' shows your ratings for various items(songs). Just as a chef uses ingredients to create a recipe, our system uses this matrix to distill your unique tastes and preferences. It then approximates this matrix by factoring it into two smaller matrices—'User Matrix' and 'Item Matrix'. These matrices highlight the underlying factors that connect your preferences to item attributes. When multiplied together, they reconstruct the original matrix, filling in the gaps with predicted ratings, suggesting new items you're likely to enjoy. This mathematical innovation powers the recommendations you see, providing transparent and personalized suggestions tailored just for you.""")
    st.image('Test.png', width=800)
    st.markdown("""
    The 'Why' Behind Recommendations: Our system uses a sophisticated algorithm called Singular Value Decomposition (SVD), which considers your past interactions to find patterns and similarities with other users and items. When it suggests something, it also provides a reason, such as 'Users like you enjoyed this item,' or 'This is similar to other items you've rated highly.'
    Our code is crafted with efficiency and maintainability in mind. We use the latest best practices to ensure that explanations are generated quickly and accurately reflect the recommendation logic.

    We chose this method over others because it strikes the right balance between transparency and complexity. It gives you enough insight to trust and understand the recommendations without overwhelming you with technical details. 

    Our recommendation engine is like a digital mind-reader, scoring an impressive 2.21 out of 10 in predicting what you'll enjoy next. This means it's good at guessing what you like, with just a tiny margin of error. It's all thanks to a clever math trick called SVD that helps us see patterns in what you and others enjoy. So, sit back and let us find you your next favorite song- our system's pretty sharp at picking songs match your taste! 😄
    """)
    
    # Title or explanation for your video
    # Display the title or explanation
    st.markdown("""
    ## A Step-by-Step Guide""")
    # Path to the video file
    video_path = "video.mp4"

    # Display the video
    st.video(video_path)

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