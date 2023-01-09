import nltk
from nltk.tokenize import word_tokenize # 워드 토크나이즈 모듈 호출

nltk.download('all', quiet=True)

text = "Friends, Romans, Countrymen, lend me your ears;."
print(word_tokenize(text))