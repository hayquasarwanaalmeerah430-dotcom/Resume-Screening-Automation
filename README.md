# Resume Screening Automation

A Python tool that automates resume filtering for intern selection.

## What it does
- Reads PDF resumes
- Extracts skills, experience and organizations using spaCy
- Matches resumes to a job description using TF-IDF and cosine similarity
- Ranks candidates and saves results to results.csv

## How to run
pip install pypdf spacy scikit-learn pandas
python -m spacy download en_core_web_sm
python resume_screening.py

## Tools
Python, spaCy, scikit-learn, pandas, pypdf
