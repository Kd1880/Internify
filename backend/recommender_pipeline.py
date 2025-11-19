from models.content_filter import ContentFilter
from models.logistic_regression import LogisticModel
from models.kmeans_model import KMeansModel
try:
	from backend.models.nlp_parser import NLPParser  # type: ignore
except Exception:
	# Fallback lightweight NLP vectorizer if NLPParser is not available
	from sklearn.feature_extraction.text import TfidfVectorizer
	class NLPParser:  # type: ignore
		def __init__(self):
			self._vectorizer = TfidfVectorizer(stop_words="english")
		def vectorize(self, texts):
			# Fit on first call if not fitted; for a single resume + many descriptions,
			# caller should fit on combined corpus externally. For simplicity, fit on input.
			return self._vectorizer.fit_transform(texts)

from backend.utils.resume_parser import extract_skills
from backend.utils.pdf_to_text import extract_text_from_pdf
from backend.db_handler import save_match, get_connection
import pandas as pd
import numpy as np
import os
from collections import Counter


def process_resume(file_path, user_id):
	"""
	Full pipeline:
	1. Convert resume PDF → text
	2. Extract skills
	3. Vectorize using NLP
	4. Run models → get scores
	5. Save results to DB
	6. Return top N internships

	Parameters
	----------
	file_path : str
		Path to the uploaded resume PDF.
	user_id : int
		ID of the user (foreign key for matches table).

	Returns
	-------
	pandas.DataFrame
		Top-N internships with columns: title, final_score, cluster.
	"""

	# Step 1 — convert resume to text
	# Read the PDF and convert to a single string.
	resume_text = extract_text_from_pdf(file_path)

	# Step 2 — extract skills (from resume_parser)
	# Basic keyword-based skill extraction; can be replaced with advanced NLP later.
	skills = extract_skills(resume_text)

	# Step 3 — load internships data
	# Source CSV containing title/description/skills for retrieval and scoring.
	internships_path = "data/internships.csv"
	if not os.path.exists(internships_path):
		raise FileNotFoundError("Internships CSV not found.")
	internships_df = pd.read_csv(internships_path)
	# Normalize column names: trim whitespace to handle accidental leading/trailing spaces
	internships_df.columns = [str(c).strip() for c in internships_df.columns]
	# Normalize optional link/url columns if present
	for link_col in ["link", "url", "apply_link", "apply_url", "application_link"]:
		if link_col in internships_df.columns:
			if link_col != "link":
				internships_df.rename(columns={link_col: "link"}, inplace=True)
			break
	# Normalize required skills list per internship for missing-skill analysis
	internships_df["required_skills"] = internships_df["required_skills"].astype(str)
	internships_df["required_skills_list"] = internships_df["required_skills"].apply(
		lambda s: [x.strip().lower() for x in s.split(",") if x.strip()]
	)
	resume_skills = [x.strip().lower() for x in (skills or []) if str(x).strip()]
	def _skill_match(lst):
		try:
			if not lst:
				return 0.0
			return 100.0 * len(set(resume_skills) & set(lst)) / max(1, len(lst))
		except Exception:
			return 0.0
	internships_df["skill_match_pct"] = internships_df["required_skills_list"].apply(_skill_match)

	# Step 4 — vectorize
	# Fit vectorizer on combined corpus ([resume] + all descriptions) to ensure a shared space.
	nlp = NLPParser()
	# For stable similarity, fit vectorizer on combined corpus then transform separately
	descriptions = internships_df["description"].astype(str).tolist()
	corpus = [resume_text] + descriptions
	corpus_vectors = nlp.vectorize(corpus)
	resume_vector = corpus_vectors[0]
	internship_vectors = corpus_vectors[1:]

	# Step 5 — load models
	# ContentFilter: cosine similarity; Logistic/KMeans: loaded from pickles when available.
	content_model = ContentFilter()
	log_model = LogisticModel()
	kmeans_model = KMeansModel()

	# Try loading persisted models; if missing, fall back gracefully
	try:
		log_model.load_model("data/model_files/logistic_model.pkl")
		logistic_available = True
	except Exception:
		logistic_available = False

	try:
		kmeans_model.load_model("data/model_files/kmeans_model.pkl")
		kmeans_available = True
	except Exception:
		# Train KMeans on-the-fly if no pickle exists
		kmeans_model.train(internship_vectors)
		kmeans_available = True

	# Step 6 — predictions
	# Similarity: [0,1], Logistic: match probability (if available), KMeans: cluster id per internship.
	similarity_scores = content_model.get_similarity(resume_vector, internship_vectors)
	logistic_probs = log_model.predict(internship_vectors) if logistic_available else np.zeros(internship_vectors.shape[0])
	clusters = kmeans_model.predict(internship_vectors) if kmeans_available else np.zeros(internship_vectors.shape[0], dtype=int)

	# Step 7 — combine score
	# Weighted blend: content similarity (0.5) + logistic probability (0.3).
	# Remaining weight can be used later for additional signals (e.g., skill overlap, recency).
	final_scores = 0.5 * similarity_scores + 0.3 * logistic_probs
	internships_df["final_score"] = final_scores
	internships_df["cluster"] = clusters

	# Step 8 — save top results in DB
	# Persist best matches for the user. internship_id uses DataFrame index as ID placeholder.
	top_results = internships_df.sort_values(by="final_score", ascending=False).head(5)
	for _, row in top_results.iterrows():
		save_match(user_id, row.name, float(row["final_score"]), int(row["cluster"]))

	# Return core columns plus optional link when available
	cols = ["company", "title", "final_score", "cluster", "skill_match_pct"]
	if "link" in top_results.columns:
		cols.append("link")
	return top_results[cols]


# optional test
if __name__ == "__main__":
	dummy_user_id = 1
	dummy_file = "data/resumes/sample_resume.pdf"
	result = process_resume(dummy_file, dummy_user_id)
	print(result)


