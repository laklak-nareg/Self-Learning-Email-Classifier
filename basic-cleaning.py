import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
nltk.download('punkt_tab')

nltk.data.path.append('C:\\Users\\nareg\\AppData\\Roaming\\nltk_data')

# Download necessary NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Example email content (replace with fetched email content)
email_content = """Hello Nareg, Thank you for your interest in employment at Axcient. We will review your application and will reach out to you if you are moved to the next stage."""

# Function to preprocess the text
def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)

    # 3. Tokenize the text
    words = word_tokenize(text)

    # 4. Remove stopwords and punctuation
    words = [word for word in words if word not in stop_words and word not in punctuation]

    # 5. Lemmatize the words (reduce words to their base forms)
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Preprocess the email content
preprocessed_content = preprocess_text(email_content)

# Output the cleaned and tokenized email
print("Preprocessed email content:", preprocessed_content)
