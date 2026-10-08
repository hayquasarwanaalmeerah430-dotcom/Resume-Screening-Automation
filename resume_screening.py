"""
Resume Screening Automation
---------------------------
1. Read every PDF resume in the "resumes" folder
2. Extract skills, experience and organizations (spaCy)
3. Compare each resume with the job description (TF-IDF + cosine similarity)
4. Rank the candidates and save the result to results.csv

Setup:
    pip install pypdf spacy scikit-learn pandas
    python -m spacy download en_core_web_sm
"""

import os
import re

import pandas as pd
import spacy
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESUME_FOLDER = "resumes"              # put all PDF resumes here
JOB_FILE = "job_description.txt"       # paste the job description here
OUTPUT_FILE = "results.csv"

nlp = spacy.load("en_core_web_sm")

SKILLS_DB = [
    "python", "java", "sql", "machine learning", "nlp",
    "excel", "communication", "project management", "aws", "react",
    "c++", "html", "css", "javascript", "data analysis",
    "power bi", "teamwork", "problem solving", "leadership",
]


# ---------- Step 1: read the resumes ----------
def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "") + " "
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- Step 2: extract information ----------
def extract_skills(text):
    text = text.lower()
    return [
        skill for skill in SKILLS_DB
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text)
    ]


def extract_experience_years(text):
    """Finds phrases like '2 years' or '3+ yrs' and returns the largest number."""
    matches = re.findall(r"(\d+)\+?\s*(?:years|year|yrs|yr)", text.lower())
    return max(int(m) for m in matches) if matches else 0


def extract_entities(text):
    doc = nlp(text[:100000])  # limit length so spaCy stays fast
    orgs = sorted({ent.text for ent in doc.ents if ent.label_ == "ORG"})
    dates = sorted({ent.text for ent in doc.ents if ent.label_ == "DATE"})
    return orgs, dates


# ---------- Step 3: match resumes to the job ----------
def tfidf_scores(job_text, resume_texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([job_text] + resume_texts)
    # row 0 = job description, remaining rows = resumes
    return cosine_similarity(matrix[0:1], matrix[1:]).flatten()


def skill_match_percent(job_skills, resume_skills):
    if not job_skills:
        return 0.0
    matched = set(job_skills) & set(resume_skills)
    return len(matched) / len(job_skills) * 100


# ---------- Step 4: run everything ----------
def main():
    with open(JOB_FILE, "r", encoding="utf-8") as f:
        job_text = clean_text(f.read())
    job_skills = extract_skills(job_text)
    print("Skills required by the job:", job_skills)

    names, texts, rows = [], [], []
    for file in sorted(os.listdir(RESUME_FOLDER)):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(RESUME_FOLDER, file)
        text = clean_text(extract_text_from_pdf(path))
        if not text:
            print(f"Skipped {file} (no readable text, maybe a scanned PDF)")
            continue

        skills = extract_skills(text)
        orgs, dates = extract_entities(text)
        names.append(file)
        texts.append(text)
        rows.append({
            "resume": file,
            "skills_found": ", ".join(skills),
            "skill_match_%": round(skill_match_percent(job_skills, skills), 1),
            "experience_years": extract_experience_years(text),
            "organizations": ", ".join(orgs[:5]),
        })

    if not rows:
        print("No resumes found. Add PDF files to the 'resumes' folder.")
        return

    scores = tfidf_scores(job_text, texts)
    for row, score in zip(rows, scores):
        row["tfidf_similarity_%"] = round(score * 100, 1)
        # final score: 60% text similarity + 40% skill match
        row["final_score"] = round(0.6 * row["tfidf_similarity_%"] + 0.4 * row["skill_match_%"], 1)

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTop candidates:")
    print(df[["rank", "resume", "skill_match_%", "tfidf_similarity_%", "final_score"]].head(10).to_string(index=False))
    print(f"\nFull results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()