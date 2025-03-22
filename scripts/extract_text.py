import os
import re
from PyPDF2 import PdfReader

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the PDF file
pdf_path = os.path.join(BASE_PATH, "new_dataset","raw", "monte.pdf")

# Extract text from pages 2 to 11 (0-based index)
with open(pdf_path, "rb") as file:
    reader = PdfReader(file)
    text = "\n".join(reader.pages[i].extract_text() for i in range(3, 28) if reader.pages[i].extract_text())

# Remove extra newlines and form a single space-separated text
text = re.sub(r'[\n\f]+', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()

# Define pattern for quoted text (curly and straight quotes)
quoted_pattern = r'“[^”]+”|"[^"]+"'

# Find all quoted texts
quoted_texts = re.findall(quoted_pattern, text)

# Create unique placeholders
placeholders = [f'<QUOTE{i}>' for i in range(len(quoted_texts))]

# Replace quoted texts with placeholders
for placeholder, quoted_text in zip(placeholders, quoted_texts):
    text = text.replace(quoted_text, placeholder)

# Split into sentences (after .!? followed by whitespace)
sentences = re.split(r'(?<=[.!?])\s+', text)

# Restore the quoted texts
for i, sentence in enumerate(sentences):
    for placeholder, quoted_text in zip(placeholders, quoted_texts):
        sentence = sentence.replace(placeholder, quoted_text)
    sentences[i] = sentence.strip()

# Filter out empty sentences
sentences = [s for s in sentences if s]

# Save to file
output_path = os.path.join(BASE_PATH, "new_dataset", "text1.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for s in sentences:
        f.write(f"{s}\n")

print("Processing complete. Sentences saved.")
