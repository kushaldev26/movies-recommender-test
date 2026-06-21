# CineMatch — Movie Recommender System

A content-based movie recommendation system built using pandas, scikit-learn (cosine similarity), FastAPI, and vanilla frontend web technologies.

## Local Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Open your terminal at the project root and run:
```bash
pip install -r backend/requirements.txt
```
*(If you are using a virtual environment, activate it before running the install command.)*

### 3. Run the Backend API
Start the FastAPI server using `uvicorn`:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
This starts the backend API on [http://127.0.0.1:8000](http://127.0.0.1:8000). You can verify if it's running by checking:
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Interactive API docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Run the Frontend
Since the frontend is built using standard HTML5, CSS3, and JavaScript, no compilation is required.

Simply locate and open the `frontend/index.html` file in any modern web browser (e.g., Double-click on `index.html` or drag it into Chrome/Edge/Firefox).
