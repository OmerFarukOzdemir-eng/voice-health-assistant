import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
import re

class NLPProcessor:
    def __init__(self):
        # Gerekli NLTK verilerini indir
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Türkçe stop words ve stemmer
        self.stop_words = set(stopwords.words('turkish'))
        self.stemmer = SnowballStemmer('turkish')
        
        # spaCy Türkçe modelini yükle
        self.nlp = spacy.load('tr_core_news_sm')
        
        # Belirti kelimeleri sözlüğü
        self.belirti_sozlugu = {
            'ağrı': ['ağrı', 'sızı', 'acı'],
            'şişlik': ['şişlik', 'şişme', 'ödem'],
            'kızarıklık': ['kızarıklık', 'kızarma'],
            'kaşıntı': ['kaşıntı', 'kaşınma'],
            'bulantı': ['bulantı', 'mide bulantısı'],
            'ateş': ['ateş', 'yüksek ateş'],
            'öksürük': ['öksürük', 'öksürme'],
            'yorgunluk': ['yorgunluk', 'halsizlik']
        }

    def _metin_temizle(self, metin):
        # Küçük harfe çevir
        metin = metin.lower()
        
        # Özel karakterleri temizle
        metin = re.sub(r'[^\w\s]', ' ', metin)
        
        # Fazla boşlukları temizle
        metin = re.sub(r'\s+', ' ', metin).strip()
        
        return metin

    def _belirtileri_bul(self, doc):
        belirtiler = []
        for token in doc:
            for belirti_grup, esanlamlar in self.belirti_sozlugu.items():
                if token.text in esanlamlar:
                    if belirti_grup not in belirtiler:
                        belirtiler.append(belirti_grup)
        return belirtiler

    def metni_isle(self, metin):
        # Metni temizle
        metin = self._metin_temizle(metin)
        
        # spaCy ile analiz
        doc = self.nlp(metin)
        
        # Belirtileri bul
        belirtiler = self._belirtileri_bul(doc)
        
        # Önemli kelimeleri çıkar
        onemli_kelimeler = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']:
                # Stemming uygula
                kelime = self.stemmer.stem(token.text)
                if (kelime not in self.stop_words and 
                    len(kelime) > 2 and 
                    kelime not in onemli_kelimeler):
                    onemli_kelimeler.append(token.text)
        
        return {
            'onemli_kelimeler': onemli_kelimeler,
            'belirtiler': belirtiler,
            'entities': [(ent.text, ent.label_) for ent in doc.ents]
        } 