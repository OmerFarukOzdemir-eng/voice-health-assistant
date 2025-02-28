from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import joblib
import json

class MLPredictor:
    def __init__(self):
        # Genişletilmiş eğitim verisi
        self.egitim_verisi = self._veri_yukle()
        self.model_egit()

    def _veri_yukle(self):
        # Gerçek bir veri seti için bu verileri bir JSON dosyasından yükleyebilirsiniz
        return {
            'sikayet': [
                'başım çok ağrıyor migren sürekli baş dönmesi var',
                'şiddetli baş ağrısı mide bulantısı ışığa hassasiyet',
                'kalbim çarpıyor göğsümde ağrı var nefes darlığı',
                'göğüs ağrısı sol kol ağrısı terleme',
                'gözüm bulanık görüyor baş ağrısı',
                'gözlerde yanma sulanma kaşıntı',
                'dişim ağrıyor şişlik var',
                'dişeti kanaması diş ağrısı',
                'eklemlerimde ağrı var hareket zorluğu',
                'dizlerimde ağrı şişlik yürüme zorluğu',
                'nefes almakta zorlanıyorum öksürük',
                'nefes darlığı göğüste baskı hırıltı',
                'cildimde döküntü var kaşıntı kızarıklık',
                'ciltte pullanma kaşıntı döküntü',
                'kulağım ağrıyor işitme problemi çınlama',
                'kulak akıntısı işitme kaybı ağrı',
                'boğazım ağrıyor öksürük ateş',
                'boğaz ağrısı yutkunma zorluğu ateş'
            ],
            'bolum': [
                'Nöroloji', 'Nöroloji',
                'Kardiyoloji', 'Kardiyoloji',
                'Göz Hastalıkları', 'Göz Hastalıkları',
                'Diş Hekimliği', 'Diş Hekimliği',
                'Ortopedi', 'Ortopedi',
                'Göğüs Hastalıkları', 'Göğüs Hastalıkları',
                'Dermatoloji', 'Dermatoloji',
                'Kulak Burun Boğaz', 'Kulak Burun Boğaz',
                'Kulak Burun Boğaz', 'Kulak Burun Boğaz'
            ]
        }

    def model_egit(self):
        # Veriyi hazırla
        X = self.egitim_verisi['sikayet']
        y = self.egitim_verisi['bolum']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # TF-IDF vektörizasyonu
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Random Forest modeli
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Model eğitimi
        self.model.fit(X_train_vec, y_train)
        
        # Model performansını değerlendir
        y_pred = self.model.predict(X_test_vec)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Model doğruluk oranı: {self.accuracy:.2f}")

    def bolum_tahmin_et(self, metin):
        # Metni vektöre dönüştür
        X_test = self.vectorizer.transform([metin])
        
        # Tahminler ve olasılıklar
        tahmin = self.model.predict(X_test)
        olasiliklar = self.model.predict_proba(X_test)
        
        # En yüksek 3 tahmini al
        en_iyi_3_index = np.argsort(olasiliklar[0])[-3:][::-1]
        sonuclar = []
        
        for idx in en_iyi_3_index:
            bolum = self.model.classes_[idx]
            olasilik = olasiliklar[0][idx]
            if olasilik >= 0.1:  # Minimum %10 güven skoru
                sonuclar.append({
                    'bolum': bolum,
                    'guven_skoru': float(olasilik)
                })
        
        return sonuclar 