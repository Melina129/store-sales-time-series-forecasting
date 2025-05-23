#  Store Sales - Time Series Forecasting (Kaggle Competition)

Bu proje, [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) yarışması için geliştirilmiştir.  
Amaç, Ekvador’daki bir perakende zincirinin mağaza ve ürün bazlı **günlük satışlarını tahmin etmektir.**

---

##  Proje Dosya Yapısı

store-sales-time-series-forecasting/
├── model.py # Ana model dosyası (LightGBM)
├── submission.csv # Kaggle submission çıktısı
├── train.csv # Eğitim verisi
├── test.csv # Test verisi
├── stores.csv # Mağaza bilgileri
├── transactions.csv # Günlük işlem sayıları
├── oil.csv # Petrol fiyatı
├── holidays_events.csv # Tatil ve özel günler
└── README.md # Bu açıklama dosyası

---

##  Kullanılan Yöntemler

- LightGBM Regressor ile zaman serisi tahmini
- Özellik mühendisliği (Feature Engineering)
  - Tarihsel değişkenler: yıl, ay, gün, hafta içi
  - Hafta sonu, ay başı, ay ortası, ay sonu gibi değişkenler
  - Tatil günleri ve tatil öncesi günler
  - Gecikmeli satışlar: `sales_lag_1`, `sales_lag_7`
- Eksik verilerin doldurulması (`ffill`) veya alternatif doldurma yöntemleri
- Label Encoding (kategorik değişkenler)
- Hiperparametre optimizasyonu (Optuna)
- Validation: son 3 ay verisiyle zaman temelli validasyon
- Submission: Kaggle formatına uygun `submission.csv` dosyası oluşturma

---

##  Skorlar

| Aşama         | RMSLE Skoru |
|---------------|-------------|
| İlk model     | 1.11842     |
| Lag features  | 1.09850 ✅  |


---

##  Gereksinimler

Python 3.9+
lightgbm
optuna
scikit-learn
pandas
numpy

### Kurulum

pip install -r requirements.txt
Not: submission.csv, train.csv, test.csv gibi dosyalar Kaggle'dan indirilmeli ve proje klasörüne eklenmelidir.

 Planlanan Geliştirmeler
Rolling mean (7-gün, 14-gün hareketli ortalama)

onpromotion_lag, transactions_lag gibi geçmiş veriye dayalı yeni değişkenler

Gün bazlı geçmiş ortalama (dow_store_family_avg)

Ensemble modeller: LightGBM + XGBoost + CatBoost

Daha ileri seviye: Darts, Temporal Fusion Transformer gibi zaman serisi modelleri

👤 Geliştirici
Melina129
