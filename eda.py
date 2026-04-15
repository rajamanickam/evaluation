import pdfplumber
from PyPDF2 import PdfReader
from langdetect import detect
import nltk
import pandas as pd
import re
from collections import Counter

# ----------------------------
# CONFIG
# ----------------------------
PDF_PATH = "pdfs/ragsample.pdf"
SCANNED_WORD_THRESHOLD = 30
KEYWORD_TOP_N = 20

nltk.download("punkt", quiet=True)

# ----------------------------
# 1. METADATA EDA
# ----------------------------
reader = PdfReader(PDF_PATH)
num_pages = len(reader.pages)
metadata = reader.metadata

print("\n=== PDF METADATA ===")
print("Total Pages:", num_pages)
print("Metadata:", metadata)

# ----------------------------
# 2. TEXT EXTRACTION & PAGE STATS
# ----------------------------
all_text = []
page_stats = []
pages_with_tables = []

with pdfplumber.open(PDF_PATH) as pdf:
    for idx, page in enumerate(pdf.pages):
        page_number = idx + 1
        text = page.extract_text() or ""
        words = text.split()

        all_text.append(text)

        page_stats.append({
            "page": page_number,
            "characters": len(text),
            "words": len(words)
        })

        # Table detection
        tables = page.extract_tables()
        if tables:
            pages_with_tables.append({
                "page": page_number,
                "tables_count": len(tables)
            })

full_text = "\n".join(all_text)
df = pd.DataFrame(page_stats)

print("\n=== PAGE LEVEL STATS ===")
print(df.describe())

# ----------------------------
# 3. SCANNED PAGE DETECTION
# ----------------------------
scanned_pages = df[df["words"] < SCANNED_WORD_THRESHOLD]

print("\n=== LIKELY SCANNED PAGES ===")
print(scanned_pages if not scanned_pages.empty else "None")

# ----------------------------
# 4. LANGUAGE DETECTION
# ----------------------------
sample_text = full_text[:1000].strip()
language = detect(sample_text) if sample_text else "Unknown"

print("\n=== LANGUAGE DETECTION ===")
print("Detected Language:", language)

# ----------------------------
# 5. SENTENCE ANALYSIS
# ----------------------------
sentences = nltk.sent_tokenize(full_text) if full_text else []
avg_sentence_length = (
    sum(len(s.split()) for s in sentences) / len(sentences)
    if sentences else 0
)

print("\n=== SENTENCE ANALYSIS ===")
print("Total Sentences:", len(sentences))
print("Average Sentence Length (words):", round(avg_sentence_length, 2))

# ----------------------------
# 6. KEYWORD FREQUENCY
# ----------------------------
words = re.findall(r"\b\w+\b", full_text.lower())
top_keywords = Counter(words).most_common(KEYWORD_TOP_N)

print("\n=== TOP KEYWORDS ===")
for word, count in top_keywords:
    print(f"{word}: {count}")

# ----------------------------
# 7. TABLE SUMMARY
# ----------------------------
print("\n=== TABLE DETECTION ===")
if pages_with_tables:
    for item in pages_with_tables:
        print(f"Page {item['page']} → {item['tables_count']} tables")
else:
    print("No tables detected")

# ----------------------------
# 8. RAG READINESS SUGGESTIONS
# ----------------------------
avg_words_per_page = df["words"].mean()

print("\n=== RAG READINESS ===")
print("Average Words Per Page:", round(avg_words_per_page, 2))

if avg_words_per_page > 500:
    chunking_suggestion = "Recursive or section-based chunking recommended"
else:
    chunking_suggestion = "Fixed-size chunking is sufficient"

print("Chunking Suggestion:", chunking_suggestion)

# ----------------------------
# 9. FINAL EDA SUMMARY
# ----------------------------
eda_summary = {
    "total_pages": num_pages,
    "total_words": int(df["words"].sum()),
    "avg_words_per_page": round(avg_words_per_page, 2),
    "scanned_pages_count": len(scanned_pages),
    "language": language,
    "pages_with_tables": len(pages_with_tables)
}

print("\n=== FINAL EDA SUMMARY ===")
for k, v in eda_summary.items():
    print(f"{k}: {v}")
