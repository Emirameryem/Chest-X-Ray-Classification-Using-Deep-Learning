import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

# Sabitler
IMG_SIZE = 224
CLASSES = ["COVID", "NORMAL", "Viral Pneumonia"]
DATA_DIR = os.path.join(os.getcwd(), "data")

def create_directory_structure():
    """Veri klasörlerini oluşturur."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for class_name in CLASSES:
        path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"[BİLGI] Klasör oluşturuldu: {path}")

# --- YARDIMCI GÖRÜNTÜ OKUMA/YAZMA FONKSİYONLARI ---
def read_image(path):
    """Unicode yolları destekleyen görüntü okuma fonksiyonu."""
    try:
        # Dosyayı binary olarak oku
        with open(path, "rb") as f:
            bytes = bytearray(f.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            # OpenCV ile decode et (RENKLİ modda)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            # BGR -> RGB çevrimi (OpenCV BGR okur, biz RGB kullanacağız)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except Exception as e:
        print(f"[HATA] Dosya okunamadı {path}: {e}")
        return None

def write_image(path, img):
    """Unicode yolları destekleyen görüntü yazma fonksiyonu."""
    try:
        # RGB -> BGR çevrimi (Kaydederken OpenCV BGR bekler)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        is_success, im_buf_arr = cv2.imencode(".png", img_bgr)
        if is_success:
            im_buf_arr.tofile(path)
        return is_success
    except Exception as e:
        print(f"[HATA] Dosya yazılamadı {path}: {e}")
        return False

def create_dummy_data(num_samples=20):
    """Eğer veri yoksa test amaçlı sahte gürültü görselleri oluşturur."""
    print(f"[DEBUG] create_dummy_data çağrıldı. Hedef: {DATA_DIR}")
    create_directory_structure()
    
    total_files = sum([len(files) for r, d, files in os.walk(DATA_DIR)])
    print(f"[DEBUG] Mevcut dosya sayısı: {total_files}")
    
    if total_files > 0:
        print("[BİLGI] Veri klasörü dolu, sahte veri oluşturulmadı.")
        return

    print("[UYARI] Gerçek veri bulunamadı. Test amaçlı 'SAHTE' veri oluşturuluyor...")
    for class_name in CLASSES:
        path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(path):
             os.makedirs(path)
             
        print(f"[DEBUG] {class_name} için veri oluşturuluyor: {path}")
        for i in range(num_samples):
            # Rastgele gürültü oluştur (Renkli - RGB)
            img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            # Rastgele şekiller
            if class_name == "COVID":
                cv2.circle(img, (100, 100), 40, (255, 0, 0), -1) # Kırmızı Daire
            elif class_name == "Viral Pneumonia":
                cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1) # Yeşil Kare
            
            fname = os.path.join(path, f"dummy_{i}.png")
            success = write_image(fname, img)
            if not success:
                 print(f"[HATA] Dosya yazılamadı: {fname}")
            
    print(f"[BAŞARILI] {num_samples*3} adet sahte görüntü oluşturuldu.")

def load_data():
    """Görüntüleri klasörlerden okur ve numpy dizisine çevirir."""
    data = []
    labels = []
    
    print("[BİLGI] Veriler yükleniyor (Transfer Learning için RGB)...")
    
    # Veri kontrolü yap
    create_dummy_data()

    for category in CLASSES:
        path = os.path.join(DATA_DIR, category)
        print(f"[DEBUG] Klasör okunuyor: {path}")
        if not os.path.exists(path):
             print(f"[HATA] Klasör bulunamadı: {path}")
             continue

        files = os.listdir(path)
        print(f"[DEBUG] {category} sınıfında {len(files)} dosya bulundu.")

        path = os.path.join(DATA_DIR, category)
        class_num = CLASSES.index(category)
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # Yeni okuma fonksiyonunu kullan (RGB Döner)
                img_array = read_image(img_path)
                
                if img_array is None:
                    print(f"[UYARI] Görüntü okunamadı (None): {img_path}")
                    continue
                    
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"[HATA] {img_name} işlenirken hata: {e}")

    print(f"[DEBUG] Toplam yüklenen veri sayısı: {len(data)}")

    if len(data) == 0:
        raise ValueError("Hiçbir veri yüklenemedi! Lütfen 'data' klasörünü kontrol edin veya silip tekrar deneyin.")

    # NumPy formatına çevir
    # RGB olduğu için kanal sayısı 3 oldu
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3) 
    
    # MobileNetV2 için özel ön işleme (Bu çok önemli!)
    # Veriyi -1 ile 1 arasına çeker
    data = tf.keras.applications.mobilenet_v2.preprocess_input(data)
    
    labels = np.array(labels)
    
    # One-hot encoding
    labels = to_categorical(labels, num_classes=len(CLASSES))

    return train_test_split(data, labels, test_size=0.2, random_state=42)

# --- GRAD-CAM GÖRSELLEŞTİRME ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv", pred_index=None):
    """
    Grad-CAM tekniği ile ısı haritası oluşturur.
    img_array: Modelin giriş formatına uygun resim (1, 224, 224, 1)
    """
    # 1. Modelin hem son conv katmanını hem de tahmin çıktısını veren yeni bir model oluştur
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Gradyanları hesapla
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Son conv katmanının çıkışına göre "class_channel"ın gradyanını al
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 3. Global Average Pooling (Gradyanların ortalamasını al)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. Ağırlıklı kombinasyon
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. ReLU ve Normalizasyon
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """Orijinal resmin üzerine heatmap bindirir."""
    # Orijinal resmi yükle
    img = cv2.imread(img_path)
    if img is None: return None
    
    # Heatmap'i 0-255 arasına çek
    heatmap = np.uint8(255 * heatmap)

    # Jet colormap kullan
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Heatmap'i orijinal resim boyutuna getir
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Orijinal resimle birleştir
    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8") # Değerleri sınırla

    # RGB'ye çevir (OpenCV BGR kullanır, Streamlit RGB ister)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

def plot_history(history, save_path="models/history_plot.png"):
    """Eğitim başarım grafiğini çizer ve kaydeder."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Eğitim Başarımı')
    plt.plot(epochs, val_acc, 'ro-', label='Doğrulama Başarımı')
    plt.title('Eğitim ve Doğrulama Başarımı')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'ro-', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()

    if not os.path.exists("models"):
        os.makedirs("models")
        
    plt.savefig(save_path)
    print(f"[BİLGI] Grafik kaydedildi: {save_path}")
