import os.path
import pickle
import base64
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import asyncio
import aiohttp
import time
from googleapiclient.errors import HttpError

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate with Gmail API and return service object."""
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

async def fetch_email_content(service, message_id):
    """Fetch a single email content and sender by ID."""
    retries = 0
    max_retries = 3
    email_body = ""
    sender = ""

    while retries < max_retries:
        try:
            msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            
            # Extract sender from headers
            for header in headers:
                if header['name'].lower() == 'from':
                    sender = header['value']
                    break

            parts = payload.get('parts', [])

            for part in parts:
                if part['mimeType'] == 'text/plain' and part.get('body', {}).get('data'):
                    body_data = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    email_body += body_data

            if email_body.strip():
                return email_body, sender
            break  # Exit retry loop on success
        except HttpError as error:
            if error.resp.status == 500:
                retries += 1
                print(f"Error 500: Internal server error, retrying... ({retries}/{max_retries})")
                await asyncio.sleep(2 ** retries)
                continue
            else:
                raise error
        except Exception as e:
            print(f"Error fetching email {message_id}: {str(e)}")
            break
    
    return email_body, sender

async def fetch_emails(service, num_emails=150, page_token=None, max_retries=3):
    """Fetch full content of emails from Gmail's Primary tab asynchronously."""
    query = 'category:primary'  # Filter for Primary tab
    emails = []
    senders = []

    retries = 0
    while retries < max_retries:
        try:
            results = service.users().messages().list(
                userId='me', maxResults=num_emails, q=query, pageToken=page_token
            ).execute()

            messages = results.get('messages', [])
            next_page_token = results.get('nextPageToken', None)

            if not messages:
                return emails, senders, next_page_token

            # Fetch email content asynchronously
            email_tasks = [fetch_email_content(service, message['id']) for message in messages]
            email_results = await asyncio.gather(*email_tasks)
            
            # Separate the email bodies and senders
            for email_body, sender in email_results:
                emails.append(email_body)
                senders.append(sender)

            break  # Exit loop on success

        except Exception as e:
            print(f"Error fetching email list: {str(e)}")
            break

    return emails, senders, next_page_token

def preprocess_text(text):
    """Preprocess the email text: clean, tokenize, remove stopwords, and lemmatize."""
    if not isinstance(text, str):
        return ""
    
    # Clean HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()  # Extract plain text
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers

    # Tokenizing the text
    words = word_tokenize(text)

    # Removing stopwords and punctuation
    words = [word for word in words if word not in stop_words and word not in punctuation]

    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Return the cleaned text as a single string
    return ' '.join(words)

async def main():
    """Main function to orchestrate fetching and processing emails."""
    service = authenticate_gmail()  # Authenticate Gmail API
    # Fetch 100-150 emails
    emails_1_to_150, senders_1_to_150, next_page_token = await fetch_emails(service, num_emails=200) ## update number of emails here
    preprocessed_emails_1_to_150 = [preprocess_text(email) for email in emails_1_to_150]    ## update number of emails here as well

    # Convert the preprocessed emails into TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(preprocessed_emails_1_to_150) ## change name of function if changed

    # Apply KMeans clustering
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X_train)

    # Output clusters with the email sender
    for i, (cluster, sender) in enumerate(zip(labels, senders_1_to_150)):
        print(f"Email {i+1} - Cluster: {cluster} - Sender: {sender}")

if __name__ == "__main__":
    asyncio.run(main())


