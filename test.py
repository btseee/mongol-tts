import csv
unique_speakers = set()
with open("/home/tsee/Downloads/cv-corpus-21.0-2025-03-14-mn/cv-corpus-21.0-2025-03-14/mn/validated.tsv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        unique_speakers.add(row['client_id'])
print(f"Хэлэгчийн тоо: {len(unique_speakers)}")