import speech_recognition as sr
from nlp_processor import NLPProcessor
from ml_predictor import MLPredictor
import warnings
import json
from datetime import datetime
import os
warnings.filterwarnings('ignore')

class SaglikAsistani:
    def __init__(self):
        print("Sağlık Asistanı başlatılıyor...")
        self.recognizer = sr.Recognizer()
        self.nlp_processor = NLPProcessor()
        self.ml_predictor = MLPredictor()
        self.session_data = []
        
    def ses_al_ve_cevir(self):
        try:
            with sr.Microphone() as mikrofon:
                print("\n🎤 Sizi dinliyorum... Lütfen şikayetinizi anlatın.")
                print("(Konuşmayı bitirmek için birkaç saniye bekleyin)")
                
                # Gürültü ayarlaması
                self.recognizer.adjust_for_ambient_noise(mikrofon, duration=1)
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.energy_threshold = 4000
                
                ses = self.recognizer.listen(mikrofon, timeout=5, phrase_time_limit=15)

            try:
                metin = self.recognizer.recognize_google(ses, language="tr-TR")
                print(f"\n🔊 Anlaşılan metin: {metin}")
                return metin
            except sr.UnknownValueError:
                print("❌ Üzgünüm, söylediklerinizi anlayamadım.")
                return None
            except sr.RequestError:
                print("❌ Google Speech Recognition servisi şu anda kullanılamıyor.")
                return None

        except Exception as e:
            print(f"❌ Bir hata oluştu: {str(e)}")
            return None

    def sonuclari_kaydet(self, metin, nlp_sonuc, tahminler):
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session = {
            'tarih': tarih,
            'metin': metin,
            'nlp_analiz': nlp_sonuc,
            'tahminler': tahminler
        }
        self.session_data.append(session)
        
        # JSON dosyasına kaydet
        os.makedirs('logs', exist_ok=True)
        with open('logs/sessions.json', 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, ensure_ascii=False, indent=2)

    def sonuclari_goster(self, nlp_sonuc, tahminler):
        print("\n📊 ANALİZ SONUÇLARI:")
        print("-" * 50)
        
        print("\n🔍 NLP Analizi:")
        print("• Önemli Kelimeler:", ", ".join(nlp_sonuc['onemli_kelimeler']))
        print("• Tespit Edilen Belirtiler:", ", ".join(nlp_sonuc['belirtiler']))
        
        print("\n🏥 Önerilen Bölümler:")
        for i, tahmin in enumerate(tahminler, 1):
            guven = tahmin['guven_skoru'] * 100
            print(f"{i}. {tahmin['bolum']} (Güven: %{guven:.1f})")
        
        print("\n⚠️ Not: Bu öneriler yalnızca yönlendirme amaçlıdır.")
        print("    Kesin teşhis için mutlaka bir doktora başvurunuz.")
        print("-" * 50)

    def calistir(self):
        while True:
            metin = self.ses_al_ve_cevir()
            if metin:
                # NLP işlemleri
                nlp_sonuc = self.nlp_processor.metni_isle(metin)
                
                # Makine öğrenimi tahminleri
                tahminler = self.ml_predictor.bolum_tahmin_et(metin)
                
                # Sonuçları göster
                self.sonuclari_goster(nlp_sonuc, tahminler)
                
                # Sonuçları kaydet
                self.sonuclari_kaydet(metin, nlp_sonuc, tahminler)
            
            # Devam etmek isteyip istemediğini sor
            devam = input("\n🔄 Başka bir şikayet belirtmek ister misiniz? (E/H): ")
            if devam.lower() != 'e':
                print("\n👋 Sağlıklı günler dileriz!")
                break

if __name__ == "__main__":
    asistan = SaglikAsistani()
    asistan.calistir() 