import re

# -------------------------------------------------------------
# Resume Parser (Simple Keyword Extraction)
# -------------------------------------------------------------
# This module extracts skills from a resume text using keyword
# matching. Later, we can replace this with spaCy or transformer-based NER.
# -------------------------------------------------------------

def extract_skills(text):
    """
    Extracts relevant skills from a given resume text.
    Strategy: keyword matching on a small curated list.
    Limitations: does not handle synonyms/phrases robustly; consider spaCy/LLM later.
    """
    skill_keywords = [
        'python', 'java', 'c++', 'machine learning', 'deep learning',
        'data analysis', 'excel', 'sql', 'tableau', 'pandas',
        'flask', 'django', 'react', 'html', 'css', 'nlp', 'pytorch',
        'tensorflow', 'transformers', 'analytics'
    ]

    text_lower = text.lower()
    found_skills = [skill for skill in skill_keywords if skill in text_lower]
    return list(set(found_skills))  # remove duplicates


def parse_resume(text):
    """
    Parses the resume text into a structured dictionary.
    Currently returns only 'skills'. Extend to include name, orgs, etc.
    """
    skills = extract_skills(text)
    return {
        "skills": skills
    }
