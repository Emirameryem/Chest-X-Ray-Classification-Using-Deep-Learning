from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def build_model(img_size=224, num_classes=3):
    """
    MobileNetV2 Transfer Learning Modeli Oluşturur.
    
    Neden MobileNetV2?
    - Hafif ve hızlıdır (Mobil cihazlar için tasarlanmıştır).
    - ImageNet veriseti (1 milyon+ resim) ile önceden eğitilmiştir.
    - Az sayıda COVID röntgeni ile bile yüksek başarı sağlar (Transfer Learning).
    
    Parametreler:
    - img_size: Giriş resim boyutu (224x224 önerilir)
    - num_classes: Sınıf sayısı (COVID, Normal, Viral Pneumonia -> 3)
    """
    
    # 1. TEMEL MODELİ YÜKLEME (Transfer Learning)
    # include_top=False: Modelin sonundaki 1000 sınıflı katmanı atıyoruz (Kendi 3 sınıfımızı ekleyeceğiz).
    # weights='imagenet': Modelin daha önce öğrendiği "görme yeteneğini" kullanıyoruz.
    print("[MODEL] MobileNetV2 ağırlıkları yükleniyor...")
    base_model = MobileNetV2(input_shape=(img_size, img_size, 3),
                             include_top=False,
                             weights='imagenet')
    
    # 2. AĞIRLIKLARI DONDURMA (Freezing)
    # Temel modelin eğitimi sırasında bozulmasını engellemek için donduruyoruz.
    # Sadece bizim ekleyeceğimiz son katmanlar eğitilecek.
    base_model.trainable = False

    # 3. KENDİ KATMANLARIMIZI EKLEME (Fine-tuning)
    inputs = Input(shape=(img_size, img_size, 3))
    
    # Resim temel modelden geçer
    x = base_model(inputs, training=False)
    
    # Global Average Pooling: Özellik haritasını (Feature Map) tek bir vektöre indirger.
    # Flatten katmanına göre daha modern ve az parametre içerir.
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    
    # Dense Katmanı: Öğrenme kapasitesini artırır.
    x = Dense(128, activation='relu')(x)
    
    # Dropout (%50): Rastgele nöronları kapatarak ezberlemeyi (overfitting) engeller.
    x = Dropout(0.5)(x)
    
    # Çıkış Katmanı: 3 Sınıf için olasılık üretir (Softmax)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Modeli Birleştirme
    model = Model(inputs, outputs, name="Covid_Detection_MobileNetV2")

    # Modeli Derleme (Compile)
    # Optimizer: Adam (En popüler ve hızlı optimize edici)
    # Loss: Categorical Crossentropy (Çok sınıflı sınıflandırma hatası)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
