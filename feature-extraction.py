from sklearn.feature_extraction.text import TfidfVectorizer


preprocessed_emails = [
    'hello nareg thank interest employment axcient review application reach moved next stage',
    'dear customer order shipped track order using following link',
    'special offer get off new collection shoes miss limited time deal'
]

# Apply TF-IDF to the preprocessed email content
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_emails)

# Convert the TF-IDF matrix to an array and print it
tfidf_matrix = X.toarray()
print("TF-IDF Matrix:")
print(tfidf_matrix)

# Print the feature names (i.e., the words corresponding to each column in the matrix)
print("\nFeature Names:")
print(vectorizer.get_feature_names_out())