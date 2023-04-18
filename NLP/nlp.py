import nltk
import re 
from nltk.stem.porter import PorterStemmer
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph="""

Greeting everyone. Today, I am here to deliver a speech on APJ Abdul Kalam. Dr APJ Abdul Kalam’s full name was Avul Pakir Zainuldeben Abdul Kalam, very few people know him by his full name as he was mostly addressed as ‘Missile Man of India’ and ‘People’s President’. He was born into a very poor family in Rameswaram on October 15, 1931. 
Since childhood, he enjoyed flying, and was equally curious to know how birds fly in the air? He was very intelligent and enjoyed reading, but his family did not have sufficient income for his school fees, so to support his education, he would wake up early in the morning and ride a bicycle 3 kilometres from home to collect newspapers and sell them. 
He was admitted to St. Joseph's College, Tiruchirapalli, and later he went on to complete a degree in physics in 1954 and then studied at the Madras Institute of Technology and graduated in aeronautical engineering in 1955. Since his childhood, Dr Abdul Alam wanted to be a pilot but couldn’t make his dream come true. He learned from his mistakes and accomplished numerous achievements in his life. After completing his degree, Abdul Kalam entered the Defense Department of India. He has been one of the key figures in building the nuclear capabilities of India.
APJ Abdul Kalam was appointed to the Indian Ministry of Defense as a Technical Advisor in 1992, after which he served with DRDO and ISRO, the country's largest organization. Considered a national hero for successful nuclear tests in 1998, a second successful nuclear test was conducted in Pokhran the same year under his supervision, after which India was included in the list of nuclear-powered nations. Abdul Kalam has been active in all space programs and development programs in India as a scientist. For developing India's Agni missile, Kalam was called 'Missile Man.'Abdul Kalam made a special technological and scientific contribution, for which, along with Bharat Ratna, India's highest honour, he was awarded the Padma Bhushan, Padam Vibhushan, etc. He was also awarded an honorary doctorate by more than 30 universities in the world for the same. 
In 2002, he was elected President of India and was the country's first scientist and non-political president. He visited many countries during his tenure as President and led India's youth through his lectures and encouraged them to move forward.  ‘My vision for India’ was a Famous Speech of APJ Abdul Kalam delivered at IIT Hyderabad in 2011, and is to this day my favourite speech. His far-reaching thinking gave India's growth a fresh path and became the youth's inspiration. Dr Abdul Kalam died on July 27, 2015, from an apparent cardiac arrest while delivering a lecture at IIM Shillong at the age of 83. He spent his entire life in service and inspiration for the nation and the youth, and his death is also while addressing the youth. His death is a never-ending loss to the country."""


ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentence = nltk.sent_tokenize(paragraph)

print(f"number of sentences {len(sentence)}")

corpus=[]


for i in range(len(sentence)):
    review=re.sub('[^a-zA-Z]',' ',sentence[i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import TfidfVectorizer

cv=TfidfVectorizer()
x=cv.fit_transform(corpus).toarray()
print(x)

print(x.shape)







 





