# ------------------------------------------------------------
# content_filter.py
# Text similarity utilities and a simple TF-IDF based recommender.
# - ContentFilter.get_similarity: cosine-similarity between a single
#   resume vector and a set of internship vectors (pre-computed).
# - get_recommendations: optional TF-IDF pipeline over CSV to get
#   top-N similar internships given extracted skills.
# ------------------------------------------------------------

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Load dataset (once globally so it’s not reloaded every time)
# ------------------------------------------------------------
# Try to load dataset relative to this file; keep optional to avoid import-time crashes
try:
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "internships.csv"))
    internship_df = pd.read_csv(DATA_PATH)
    # Normalize column names expected downstream
    if "skills_required" not in internship_df.columns and "required_skills" in internship_df.columns:
        internship_df = internship_df.rename(columns={"required_skills": "skills_required"})
except Exception:
    internship_df = pd.DataFrame(columns=["title", "description", "skills_required"])

# Combine all useful text columns into one “combined” column
# This helps TF-IDF learn relationships between title, description, and skills
if not internship_df.empty:
    internship_df["combined"] = (
        internship_df["title"].fillna('') + " " +
        internship_df["description"].fillna('') + " " +
        internship_df["skills_required"].fillna('')
    )
else:
    internship_df["combined"] = []

# ------------------------------------------------------------
# Create TF-IDF matrix
# ------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english")  # remove common words (like “and”, “the”)
tfidf_matrix = tfidf.fit_transform(internship_df["combined"]) if not internship_df.empty else None

class ContentFilter:
    """
    Provides cosine-similarity between a single resume vector and
    a matrix of internship vectors. Expects pre-computed vectors.
    """
    def get_similarity(self, resume_vector, internship_vectors):
        """
        Computes cosine similarity between one resume vector (1 x d)
        and many internship vectors (n x d). Returns a 1D array of
        length n with similarity scores in [0, 1].
        """
        return cosine_similarity(resume_vector, internship_vectors).flatten()


def get_recommendations(extracted_skills, top_n=5):
    """
    Input  : extracted_skills → list of skills (from NLP)
    Output : Top N internship recommendations (DataFrame rows)
    Notes  : Uses this module's TF-IDF pipeline over internships.csv.
             This is a legacy helper and not used by the new pipeline.
    """
    # Convert skills list to string for vectorization
    resume_text = " ".join(extracted_skills)

    # Vectorize resume text using the same TF-IDF model
    resume_vector = tfidf.transform([resume_text]) if tfidf_matrix is not None else None

    # Compute cosine similarity between resume and all internships
    if tfidf_matrix is None or resume_vector is None:
        return []
    similarity_scores = cosine_similarity(resume_vector, tfidf_matrix).flatten()

    # Get top N most similar internships (highest cosine values)
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    # Return corresponding rows as a list of dictionaries
    cols = [c for c in ["id", "title", "company", "skills_required"] if c in internship_df.columns]
    results = internship_df.iloc[top_indices][cols]
    return results.to_dict(orient="records")