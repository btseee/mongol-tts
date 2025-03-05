import csv
import string

characters = set()
punctuations = set()

with open('data/metadata.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        transcription = row[1].lower()  # Convert to lowercase for simplicity
        for char in transcription:
            if char.isalpha():
                characters.add(char)
            elif char in string.punctuation or char.isspace():
                punctuations.add(char)

print("Characters:", sorted(characters))
print("Punctuations:", sorted(punctuations))