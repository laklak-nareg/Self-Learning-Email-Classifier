import nltk
nltk.download('punkt_tab')
nltk.data.path.append('C:\\Users\\nareg\\AppData\\Roaming\\nltk_data')

from nltk.tokenize import word_tokenize

text = "Hello, Nareg! How are you?"
tokens = word_tokenize(text)
print(tokens)