import os
import numpy as np
import tensorflow as tf
from model import build_model
from utils import load_data, plot_history, CLASSES, IMG_SIZE

# Modeli eğitme ve kaydetme
def train():
    print("========================================")
    print("   COVID-19 TESPİT MODELİ EĞİTİMİ      ")
    print("========================================")

    # 1. Veriyi Yükle
    (X_train, X_test, y_train, y_test) = load_data()
    
    print(f"[BİLGI] Eğitim Verisi: {X_train.shape}")
    print(f"[BİLGI] Test Verisi: {X_test.shape}")

    # 2. Modeli Oluştur
    print("[BİLGI] Model oluşturuluyor...")
    model = build_model(img_size=IMG_SIZE, num_classes=len(CLASSES))
    model.summary()

    # 3. Model Eğitimi (Training)
    # ---------------------------------------------------------
    # Epoch: Veri setinin model üzerinden kaç kez geçeceği.
    # Batch Size: Her adımda kaç resmin işleneceği.
    
    EPOCHS = 25 # Küçük veri seti olduğu için 25 tur eğitiyoruz.
    BATCH_SIZE = 4 # RAM dostu olması için küçük tuttuk.
    
    # Veri Artırma (Data Augmentation)
    # Elimizdeki az sayıdaki resmi (12 adet) çoğaltarak modelin ezberlemesini önlüyoruz.
    # Resimler her turda rastgele döndürülür, kaydırılır veya büyütülür.
    print("[BİLGI] Veri Artırma (Data Augmentation) hazırlanıyor...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,      # 20 derece döndür
        width_shift_range=0.2,  # Sağa sola kaydır
        height_shift_range=0.2, # Yukarı aşağı kaydır
        shear_range=0.2,        # Yamult
        zoom_range=0.2,         # Yakınlaştır
        horizontal_flip=True,   # Aynala
        fill_mode='nearest'     # Boşlukları doldur
    )
    
    print(f"[BİLGI] Eğitim başlıyor ({EPOCHS} Epoch)...")
    print("NOT: Eğitim sırasında 'accuracy' (başarı) artmalı, 'loss' (hata) azalmalıdır.")
    
    # Modeli ImageDataGenerator akışı ile eğitiyoruz
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS)

    # 4. Modeli Kaydetme
    # ---------------------------------------------------------
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model.save("models/covid_model.h5")
    print("[BAŞARILI] Eğitilen model 'models/covid_model.h5' dosyasına kaydedildi.")

    # 5. Grafikleri Çiz
    plot_history(history)

if __name__ == "__main__":
    train()
