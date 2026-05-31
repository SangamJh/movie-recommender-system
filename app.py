import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MovieMatch",
    page_icon="🎬",
    layout="wide"
)

# =========================================================
# COLORS
# =========================================================
background = "#0f172a"
card_bg = "#1e293b"
text_color = "#ffffff"

# =========================================================
# CUSTOM CSS (YOUR UI - UNCHANGED)
# =========================================================
st.markdown(f"""
<style>

[data-testid="stAppViewContainer"] {{
    background-color: {background};
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stSidebar"] {{
    background-color: {card_bg};
}}

html, body, [class*="css"] {{
    color: {text_color};
    font-family: 'Poppins', sans-serif;
}}

.main {{
    background-color: {background};
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

.title {{
    text-align: center;
    font-size: 65px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}}

.subtitle {{
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 40px;
}}

.movie-card {{
    background: linear-gradient(145deg, #1e293b, #334155);
    border-radius: 20px;
    padding: 15px;
    text-align: center;
    transition: all 0.4s ease;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}}

.movie-card:hover {{
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 15px 30px rgba(130,69,236,0.5);
}}

.movie-title {{
    font-size: 18px;
    font-weight: bold;
    margin-top: 15px;
    color: white;
}}

.stButton > button {{
    width: 100%;
    height: 55px;
    background: linear-gradient(90deg, #8245ec, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}}

.stButton > button:hover {{
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(130,69,236,0.8);
}}

.stSelectbox label {{
    font-size: 20px;
    color: white;
    font-weight: bold;
}}

.about-text {{
    color: white;
    font-size: 15px;
    line-height: 1.8;
}}

.about-title {{
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 10px;
}}

.footer-title {{
    text-align: center;
    color: white;
    margin-bottom: 25px;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA (NO PICKLE)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv", encoding="latin1")
    df = df[["id", "title", "overview"]]
    df["overview"] = df["overview"].fillna("")
    return df

movies = load_data()

# =========================================================
# BUILD SIMILARITY (NO PICKLE)
# =========================================================
@st.cache_resource
def build_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["overview"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = build_similarity(movies)

# =========================================================
# API KEYS
# =========================================================
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# =========================================================
# FETCH POSTER (FIXED + SAFE)
# =========================================================
def fetch_poster(movie_id, movie_title):

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=10).json()

        poster = data.get("poster_path")
        if poster:
            return "https://image.tmdb.org/t/p/w500/" + poster
    except:
        pass

    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        data = requests.get(url, timeout=10).json()

        results = data.get("results", [])
        if results:
            poster = results[0].get("poster_path")
            if poster:
                return "https://image.tmdb.org/t/p/w500/" + poster
    except:
        pass

    try:
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
        data = requests.get(url, timeout=10).json()

        poster = data.get("Poster")
        if poster and poster != "N/A":
            return poster
    except:
        pass

    return "https://via.placeholder.com/500x750?text=No+Poster"

# =========================================================
# RECOMMEND FUNCTION (FIXED)
# =========================================================
def recommend(movie):

    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        title = movies.iloc[i[0]].title

        names.append(title)
        posters.append(fetch_poster(movie_id, title))

    return names, posters

# =========================================================
# UI (SAME DESIGN)
# =========================================================
st.markdown('<div class="title">🎬 MovieMatch</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Discover movies with an AI-powered recommendation system</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR ABOUT
# =========================================================
st.sidebar.markdown(
    """
    <div class="about-title"> About</div>

    <div class="about-text">
    MovieMatch is a modern AI-powered movie recommendation system.
    It uses Machine Learning and Cosine Similarity to suggest similar movies based on story overview.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# =========================================================
# FEEDBACK
# =========================================================
st.sidebar.markdown("<h2 style='color:white;'>💬 Feedback</h2>", unsafe_allow_html=True)

feedback = st.sidebar.text_area("Share your feedback")

if st.sidebar.button("Submit Feedback"):
    if feedback.strip():
        st.sidebar.success("Thanks for your feedback ❤️")
    else:
        st.sidebar.warning("Please write something")

# =========================================================
# MOVIE SELECTOR
# =========================================================
selected_movie = st.selectbox(
    "Search Your Favorite Movie",
    movies["title"].values
)

# =========================================================
# RECOMMEND BUTTON
# =========================================================
if st.button("Recommend Movies"):

    with st.spinner("Finding amazing movies for you..."):

        names, posters = recommend(selected_movie)

        st.markdown("## Recommended Movies")

        cols = st.columns(5)

        for i in range(5):
            with cols[i]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                st.image(posters[i], use_container_width=True)
                st.markdown(f'<div class="movie-title">{names[i]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER (UNCHANGED)
# =========================================================
st.markdown("---")

st.markdown('<h2 class="footer-title">🌐 Connect With Me</h2>', unsafe_allow_html=True)

space1, col1, col2, col3, col4, space2 = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1])

with col1:
    st.link_button("GitHub", "https://github.com/SangamJh", use_container_width=True)

with col2:
    st.link_button("LinkedIn", "https://www.linkedin.com/in/sangamjha86", use_container_width=True)

with col3:
    st.link_button("Twitter", "https://x.com/SangamJha86", use_container_width=True)

with col4:
    st.link_button("Portfolio", "https://sangamjhaportfolilo.vercel.app/", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:gray;'>© 2026 Sangam Jha. All rights reserved.</p>",
    unsafe_allow_html=True
)