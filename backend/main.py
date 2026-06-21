import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Check path relative to main.py first (parent directory of main.py) and then local directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load env variables from .env file (search project root first, then backend directory)
for directory in [parent_dir, current_dir, os.getcwd()]:
    env_path = os.path.join(directory, '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = FastAPI(title="CineMatch Recommendation API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_file(filename):
    # Try parent directory (project root)
    path1 = os.path.join(parent_dir, filename)
    if os.path.exists(path1):
        return path1
    # Try current directory (backend/)
    path2 = os.path.join(current_dir, filename)
    if os.path.exists(path2):
        return path2
    # Try current working directory
    if os.path.exists(filename):
        return filename
    return None

def download_from_google_drive(file_id, dest_path):
    print(f"Downloading similarity.pkl from Google Drive (ID: {file_id}) to {dest_path}...")
    import gdown
    # gdown automatically handles virus warnings and large file downloads from Google Drive
    gdown.download(id=file_id, output=dest_path, quiet=False)
    print("Download complete!")

movies_path = find_file('movies.pkl')
similarity_path = find_file('similarity.pkl')

if not similarity_path:
    # Try downloading if environment variable is set
    drive_id = os.getenv("SIMILARITY_DRIVE_ID")
    if drive_id:
        target_path = os.path.join(parent_dir, 'similarity.pkl')
        try:
            download_from_google_drive(drive_id, target_path)
            similarity_path = target_path
        except Exception as e:
            print(f"Error downloading similarity.pkl: {str(e)}")

if not movies_path or not similarity_path:
    raise FileNotFoundError(
        f"Could not find movies.pkl or similarity.pkl. Please ensure they exist in the root of the project."
    )

print(f"Loading movies from: {movies_path}")
with open(movies_path, 'rb') as f:
    movies = pickle.load(f)

print(f"Loading similarity matrix from: {similarity_path}")
with open(similarity_path, 'rb') as f:
    similarity = pickle.load(f)


class RecommendRequest(BaseModel):
    movie_title: str
    n: int = 5


@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def read_root():
    frontend_path = os.path.join(parent_dir, 'frontend', 'index.html')
    if not os.path.exists(frontend_path):
        frontend_path = os.path.join(current_dir, 'frontend', 'index.html')
    try:
        with open(frontend_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>CineMatch API is running</h1><p>Error serving UI: {str(e)}</p>",
            status_code=200
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/movies")
def get_movies():
    try:
        # Return sorted list of all unique titles as a JSON array (for autocomplete)
        titles = sorted(movies['title'].dropna().unique().tolist())
        return titles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving movies: {str(e)}")


@app.post("/recommend")
def recommend_endpoint(request: RecommendRequest):
    movie_title = request.movie_title.strip()
    n = request.n
    
    # Search for movie matching case-insensitively for a friendlier UX, but prioritize exact case match if possible
    exact_match = movies[movies['title'] == movie_title]
    if not exact_match.empty:
        idx = exact_match.index[0]
    else:
        case_insensitive_match = movies[movies['title'].str.lower() == movie_title.lower()]
        if not case_insensitive_match.empty:
            idx = case_insensitive_match.index[0]
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Movie '{movie_title}' not found. Try another title."
            )
            
    try:
        # Get distances sorted in descending order, exclude the query movie itself
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        results = []
        for i, score in distances[1:n+1]:
            movie_id = int(movies.iloc[i].movie_id)
            title = movies.iloc[i].title
            poster_path = None
            
            if TMDB_API_KEY:
                try:
                    import requests
                    # Query TMDB API with 1-second timeout to keep the app responsive
                    tmdb_res = requests.get(
                        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}", 
                        timeout=1.0
                    )
                    if tmdb_res.status_code == 200:
                        data = tmdb_res.json()
                        raw_path = data.get("poster_path")
                        if raw_path:
                            poster_path = f"https://image.tmdb.org/t/p/w300{raw_path}"
                except Exception as e:
                    print(f"Error fetching TMDB poster for movie {movie_id}: {str(e)}")
            
            results.append({
                "title": title,
                "movie_id": movie_id,
                "score": round(float(score), 4),
                "poster_path": poster_path
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
