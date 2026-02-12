# COVID-19 Tespit Sistemi - Kurulum ve Kullanım

Bu proje, yapay sinir ağları kullanarak Akciğer röntgenlerinden COVID-19, Normal ve Viral Pnömoni teşhisi koyar.

## Kurulum

1.  **Gerekli Kütüphaneleri Yükleyin**:
    Terminalde aşağıdaki komutu çalıştırın:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım Adımları

### 1. Modeli Eğitme (İlk Adım)
Önce modeli eğitmeniz gerekir. Eğer veri klasörünüz boşsa, sistem test amaçlı otomatik olarak "Sahte/Dummy" görüntüler oluşturur ve modeli bunlarla eğitir.

```bash
python src/train.py
```
Bu işlem sonunda `models/covid_model.h5` dosyası oluşturulacaktır.

### 2. Arayüzü Çalıştırma
Model eğitildikten sonra, web arayüzünü başlatmak için:

```bash
streamlit run src/app.py
```
Bu komut tarayıcınızı açacak ve uygulamayı başlatacaktır.

## Gerçek Veri ile Çalışma (Veri Seti Entegrasyonu)

Modeli gerçek dünyada kullanmak için gerçek röntgen görüntülerine ihtiyacınız var.

### 1. Veri Setini İndirme
En iyi ve popüler kaynak **Kaggle**'dır.
🔗 **Link:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Bu linke gidin ve "Download" butonuna basarak dosyayı indirin (Yaklaşık 700MB - 1GB).

### 2. Klasöre Yerleştirme
İndirdiğiniz ZIP dosyasını açın. İçinde `COVID`, `Normal`, `Viral Pneumonia` klasörlerini göreceksiniz.
Bu klasörlerdeki resimleri, projenizin `data` klasörüne kopyalayın.

Doğru yapı şöyle olmalıdır:
```text
Covid-tespiti/
└── data/
    ├── COVID/            (İçinde covid-1.png, covid-2.png...)
    ├── NORMAL/           (İçinde normal-1.png, normal-2.png...)
    └── Viral Pneumonia/  (İçinde pneumonia-1.png...)
```

### 3. Modeli Gerçek Veriyle Eğitme
Verileri attıktan sonra terminali açın ve şu komutu çalıştırın:

```bash
python src/train.py
```
Bu işlem bilgisayarınızın hızına göre 10-30 dakika sürebilir. Yeni model `models/covid_model.h5` dosyasına kaydedilecektir.
