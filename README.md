# Self-Learning Email Classifier
This project is an unsupervised learning-based email classifier that fetches emails from a Gmail account using the Gmail API, preprocesses the text, and clusters the emails using a machine learning algorithm (KMeans clustering). It automatically organizes emails into different categories based on their content, helping users better manage their inbox.

## Features
Fetches emails from the Primary inbox using the Gmail API.
Preprocesses email content by cleaning, tokenizing, and lemmatizing text.
Uses TF-IDF for feature extraction to represent the email text.
Clusters emails into distinct groups using KMeans clustering.
Outputs the cluster for each email, which helps categorize the emails into groups like work, personal, or newsletter.

# Dependencies
The project relies on the following Python libraries:

google-auth-oauthlib: For authenticating with the Gmail API.
google-auth-httplib2: For accessing Google services.
google-api-python-client: For interacting with the Gmail API.
nltk: For natural language processing (tokenization, stopwords, and lemmatization).
scikit-learn: For TF-IDF vectorization and KMeans clustering.
joblib: For model serialization and deserialization (optional).


## Installation
1. Clone the repository:
git clone https://github.com/laklak-nareg/Self-Learning-Email-Classifier.git
cd Self-Learning-Email-Classifier

2. Set Up Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


3. Install Dependencies
pip install -r requirements.txt


4. Set Up Gmail API Credentials
To authenticate with the Gmail API, you'll need to set up API credentials:

a.Go to the Google Cloud Console.
b.Create a new project and enable the Gmail API.
c.Create OAuth 2.0 credentials and download the credentials.json file.
d.Save the credentials.json file in the project directory.

5. Create a .gitignore File (Optional)
Add venv/ to the .gitignore file to prevent your virtual environment from being uploaded to the repository:

venv/


## Usage
### Fetch and Cluster Emails
1) Run the script to authenticate with Gmail, fetch emails, preprocess them, and cluster them into groups {example (0= work, 1= personal, 2= newsletter, etc...)}

python classifier-simple.py

2) The script will authenticate with Gmail, fetch the first 100 emails, and cluster them into distinct groups using KMeans.

3) After fetching the emails, the output will show which cluster each email belongs to.


## Example Output:
Email 1 - Cluster: 0
Email 2 - Cluster: 1
Email 3 - Cluster: 0
...
Email 50 - Cluster: 2

## Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!


Contact:
If you have any questions or suggestions, please feel free to contact:

Author : Nareg Laklakian
email: nareglaklakian@gmail.com
