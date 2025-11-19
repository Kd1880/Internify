import spacy
from transformers import pipeline

# ------------------------------------------------------------
# Load NLP models
# ------------------------------------------------------------

# 1️⃣ spaCy small English model → good for tokenization, POS tagging, and basic entities
nlp_spacy = spacy.load("en_core_web_sm")

# 2️⃣ HuggingFace Transformers NER pipeline
#     Uses a pretrained BERT model to detect entities like ORG, PERSON, LOC etc.
ner_model = pipeline(
    "ner",                               # task type: named-entity recognition
    model="dslim/bert-base-NER",         # pretrained model name
    aggregation_strategy="simple"        # merges sub-tokens (e.g. 'New' + 'York' → 'New York')
)

# ------------------------------------------------------------
# Simple list of common skills
# (Later we can load this from an external CSV or dataset)
# ------------------------------------------------------------
COMMON_SKILLS = [
    "python", "java", "c++", "sql", "flask", "react", "machine learning",
    "data analysis", "javascript", "tensorflow", "nlp", "html", "css",
    "deep learning", "communication", "teamwork"
]

# ------------------------------------------------------------
# Function: extract_entities(text)
# ------------------------------------------------------------
def extract_entities(text):
    """
    Extracts important information from resume text using spaCy + HF NER.

    Returns a dictionary with:\n
    - PERSON, ORG, GPE, DATE lists from spaCy/transformers\n
    - SKILLS from a simple keyword list\n

    Note: The recommender pipeline currently uses a TF-IDF fallback class\n
    when NLPParser is not present. This module is for richer NLP expansion.
    """
    # Run spaCy on the text → returns a Doc object
    doc = nlp_spacy(text)

    # Initialize dictionary for storing extracted data
    entities = {
        "PERSON": [],      # candidate name(s)
        "ORG": [],         # companies or universities
        "EDUCATION": [],   # (we’ll add manual rules later)
        "EXPERIENCE": [],  # work experience keywords
        "SKILLS": [],      # skills list
        "GPE": [],         # geopolitical entities (locations)
        "DATE": [],        # dates
    }

    # --------------------------------------------------------
    # A. Named-entity extraction with spaCy
    # --------------------------------------------------------
    for ent in doc.ents:  # loop over detected entities
        # ent.label_ gives entity type (e.g. PERSON, ORG, DATE, etc.)
        if ent.label_ in ["ORG", "PERSON", "GPE", "DATE"]:
            entities[ent.label_].append(ent.text)

    # --------------------------------------------------------
    # B. Extra entity detection with HuggingFace NER model
    # (BERT is often more accurate than spaCy on named entities)
    # --------------------------------------------------------
    hf_entities = ner_model(text)
    for e in hf_entities:
        # We care mainly about organizations and person names here
        if e['entity_group'] in ["ORG", "PER"]:
            entities["ORG"].append(e['word'])

    # --------------------------------------------------------
    # C. Skill extraction (simple keyword-based)
    # --------------------------------------------------------
    skills_found = []
    text_lower = text.lower()  # lowercase text for easy matching
    for skill in COMMON_SKILLS:
        if skill in text_lower:
            skills_found.append(skill)

    # Use set() to remove duplicates
    entities["SKILLS"] = list(set(skills_found))

    # Return all extracted information
    return entities
