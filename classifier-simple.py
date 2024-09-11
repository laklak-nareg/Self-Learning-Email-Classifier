import os.path
import pickle
import base64
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
from googleapiclient.errors import HttpError
import joblib
from sklearn.cluster import KMeans


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initializing lemmatizer and defining stopwords and punctuatians
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# gmail api scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate with Gmail API and return service object"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('gmail', 'v1', credentials=creds)
    return service

def fetch_emails(service, num_emails=50, page_token=None, max_retries=3):
    """Fetch full content of emails from Gmail's Primary tab with retry on errors."""
    query = 'category:primary'  # Filter for Primary tab
    results = service.users().messages().list(userId='me', maxResults=num_emails, q=query, pageToken=page_token).execute()
    messages = results.get('messages', [])
    next_page_token = results.get('nextPageToken', None)

    emails = []
    for message in messages:
        retries = 0
        while retries < max_retries:
            try:
                msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
                payload = msg.get('payload', {})
                parts = payload.get('parts', [])

                email_body = ""
                for part in parts:
                    if part['mimeType'] == 'text/plain' and part.get('body', {}).get('data'):
                        body_data = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        email_body += body_data

                if email_body.strip():  # Only append non-empty email bodies
                    emails.append(email_body)
                break  # Exit retry loop on success

            except HttpError as error:
                if error.resp.status == 500:
                    retries += 1
                    print(f"Error 500: Internal server error, retrying... ({retries}/{max_retries})")
                    time.sleep(2)  # Wait 2 seconds before retrying
                else:
                    raise error  # Raise other errors
            except Exception as e:
                print(f"Error fetching email {message['id']}: {str(e)}")
                break  # Stop retrying if it's a different error

    return emails, next_page_token

def preprocess_text(text):
    """Preprocess the email text: clean, tokenize, remove stopwords, and lemmatize."""

    if not isinstance(text,str):
        return ""
    # clean html tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text() # extracting plain text
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text) # remove URLs
    text = re.sub(r'\dr+','',text) #remove numbers
    # tokenizing the text
    words = word_tokenize(text)
    #removing the stopwords and punctuation
    words = [word for word in words if word not in stop_words and word not in punctuation]
    #lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    # return the cleaned text as a single string
    return ' '.join(words)

# Step 1: Authenticate Gmail API
service = authenticate_gmail()

# Step 2: Fetch the first 50 emails and manually categorize them
emails_1_to_50, next_page_token = fetch_emails(service, num_emails=50)
emails_1_to_100, next_page_token = fetch_emails(service, num_emails=100)

# Manually label the first 50 emails (0 = work, 1 = personal, 2 = newsletter) // this is for supervised learning
# labels = [
#     0,0,0,0,2,1,1,1,1,0,0,2,0,1,1,0,0,0,0,0,1,1,0,0,2,1,0,0,1,0,1,1,1,0,0,0,0
# ]

# Preprocess the first 50 emails
preprocessed_emails_1_to_50 = [preprocess_text(email) for email in emails_1_to_50]
preprocessed_emails_1_to_100 = [preprocess_text(email) for email in emails_1_to_100]



# this is to ensure that the size of the data and the labels mathces
# if len(preprocessed_emails_1_to_50) != len(labels):
#     raise ValueError (f"Mismatch between number of emails ({len(preprocessed_emails_1_to_50)}) and labels ({len(labels)})") 

# Convert the preprocessed emails into TF-IDF features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(preprocessed_emails_1_to_50)

# applying kmeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X_train)

clusters = kmeans.labels_

for i, cluster in enumerate(clusters):
    print(f"Email {i+1} - Cluster: {cluster}")


emails_101_to_150, _ = fetch_emails(service, num_emails=100, page_token=next_page_token)
preprocessed_emails_101_to_150 = [preprocess_text(email) for email in emails_101_to_150]


# # Step 3: Train the Naive Bayes model
# model = MultinomialNB()
# model.fit(X_train, labels)

# # Optional: Evaluate the model's training accuracy on the first 50 emails
# y_train_pred = model.predict(X_train)
# training_accuracy = accuracy_score(labels, y_train_pred)
# print(f"Training Accuracy on First 50 Emails: {training_accuracy * 100:.2f}%")

# # Step 4: Fetch the next 50 emails (emails 51 to 100) using the nextPageToken
# emails_51_to_100, _ = fetch_emails(service, num_emails=50, page_token=next_page_token)

# # Preprocess the next 50 emails
# preprocessed_emails_51_to_100 = [preprocess_text(email) for email in emails_51_to_100]

# Transform the new emails using the same TF-IDF vectorizer
X_new = vectorizer.transform(preprocessed_emails_101_to_150)

# Step 5: Predict the categories for the new emails (emails 51 to 100)
predicted_categories = kmeans.predict(X_new)

# Output the predicted categories for the next 50 emails
for i, category in enumerate(predicted_categories):
    print(f"Email {i + 101} - Predicted Category: {category}")

# -----------------------------------------------------------------------------------------------------------------------

# import os.path
# import pickle
# import base64
# from bs4 import BeautifulSoup
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# import re
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')

# # Initializing lemmatizer and defining stopwords and punctuation
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# punctuation = set(string.punctuation)

# # Gmail API scope
# SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# def authenticate_gmail():
#     """Authenticate with Gmail API and return service object"""
#     creds = None
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#     service = build('gmail', 'v1', credentials=creds)
#     return service

# def fetch_emails(service, num_emails=50, page_token=None):
#     """Fetch full content of emails from Gmail's Primary tab."""
#     query = 'category:primary'  # This query filters for the Primary tab
#     results = service.users().messages().list(userId='me', maxResults=num_emails, q=query, pageToken=page_token).execute()
#     messages = results.get('messages', [])
#     next_page_token = results.get('nextPageToken', None)
#     emails = []
#     for message in messages:
#         msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
#         payload = msg.get('payload', {})
#         parts = payload.get('parts', [])
#         email_body = ""
#         for part in parts:
#             if part['mimeType'] == 'text/plain' and part.get('body', {}).get('data'):
#                 body_data = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
#                 email_body += body_data
#         if email_body.strip():  # Only append non-empty email bodies
#             emails.append(email_body)
#     return emails, next_page_token


# def preprocess_text(text):
#     """Preprocess the email text: clean, tokenize, remove stopwords, and lemmatize."""
#     if not isinstance(text, str):
#         return ""
#     # Clean HTML tags
#     soup = BeautifulSoup(text, 'html.parser')
#     text = soup.get_text()  # Extract plain text
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     # Tokenize the text
#     words = word_tokenize(text)
#     # Remove stopwords and punctuation
#     words = [word for word in words if word not in stop_words and word not in punctuation]
#     # Lemmatize the words
#     words = [lemmatizer.lemmatize(word) for word in words]
#     # Return the cleaned text as a single string
#     return ' '.join(words)

# # Step 1: Authenticate Gmail API
# service = authenticate_gmail()

# # Step 2: Fetch the first 50 emails
# emails_1_to_50, next_page_token = fetch_emails(service, num_emails=50)

# # Check which emails pass the preprocessing step
# for i, email in enumerate(emails_1_to_50):
#     processed_email = preprocess_text(email)
#     if not processed_email.strip():
#         print(f"Email {i+1} is empty after preprocessing and will be skipped.")
#     else:
#         print(f"Email {i+1} passed preprocessing: {processed_email[:100]}...")  # Print first 100 characters of the preprocessed email

# Uncomment the following sections on the second run

# # Manually label the first 50 emails (0 = work, 1 = personal, 2 = newsletter)
# labels = [
#     0,0,0,0,2,1,1,1,1,0,0,2,0,1,1,0,0,0,0,0,1,1,0,0,2,1,0,0,1,0,1,1,1,0,0,0,0
# ]

# # Preprocess the first 50 emails
# preprocessed_emails_1_to_50 = [preprocess_text(email) for email in emails_1_to_50]

# if len(preprocessed_emails_1_to_50) != len(labels):
#     raise ValueError(f"Mismatch between number of emails ({len(preprocessed_emails_1_to_50)}) and labels ({len(labels)})")

# # Convert the preprocessed emails into TF-IDF features
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(preprocessed_emails_1_to_50)

# # Step 3: Train the Naive Bayes model
# model = MultinomialNB()
# model.fit(X_train, labels)

# # Optional: Evaluate the model's training accuracy on the first 50 emails
# y_train_pred = model.predict(X_train)
# training_accuracy = accuracy_score(labels, y_train_pred)
# print(f"Training Accuracy on First 50 Emails: {training_accuracy * 100:.2f}%")

# # Step 4: Fetch the next 50 emails (emails 51 to 100) using the nextPageToken
# emails_51_to_100, _ = fetch_emails(service, num_emails=50, page_token=next_page_token)

# # Preprocess the next 50 emails
# preprocessed_emails_51_to_100 = [preprocess_text(email) for email in emails_51_to_100]

# # Transform the new emails using the same TF-IDF vectorizer
# X_new = vectorizer.transform(preprocessed_emails_51_to_100)

# # Step 5: Predict the categories for the new emails (emails 51 to 100)
# predicted_categories = model.predict(X_new)

# # Output the predicted categories for the next 50 emails
# for i, category in enumerate(predicted_categories):
#     print(f"Email {i + 51} - Predicted Category: {category}")

