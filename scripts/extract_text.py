import re
from pdfminer.high_level import extract_text

# Path to the PDF file
pdf_path = "data/raw/100jil.pdf"

# Extract text from pages 2 to 11
text = extract_text(pdf_path, page_numbers=range(2, 11))

# Remove extra newlines
text = re.sub(r'[\n\f]+', ' ', text)

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

# Limit to first 180 sentences
sentences = sentences[:180]

# Save to file
with open("data/processed/text.txt", "w", encoding="utf-8") as f:
    for i, s in enumerate(sentences):
        f.write(f"{s}\n")

print("Processing complete. Sentences saved.")