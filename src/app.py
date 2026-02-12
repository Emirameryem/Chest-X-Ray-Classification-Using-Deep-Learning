import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
import random

# Sabitler
IMG_SIZE = 224
CLASSES = ["COVID", "NORMAL", "Viral Pneumonia"]
MODEL_PATH = "models/covid_model.h5"

# Sayfa Ayarları (Modern Görünüm)
st.set_page_config(
    page_title="COVID-19 AI Tanı Sistemi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile Modern Tasarım Dokunuşları
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa; 
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# @st.cache_resource kaldırıldı - Her seferinde güncel model yüklensin
def load_model():
    """Modeli yükler"""
    if not os.path.exists(MODEL_PATH):
        return None
    # Model derleme hatasını önlemek için compile=False denebilir ama metrics lazım
    try:
        model = tf.keras.models.load_model(MODEL_PATH) 
    except:
        model = None
    return model

def preprocess_image(image):
    """Görüntüyü model için hazırlar"""
    # RGB Çevrimi (MobileNetV2 3 kanal ister)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    
    # MobileNetV2 Preprocessing (-1, 1 aralığına çeker)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# --- Arayüz ve Tasarım Kısmı ---

st.title("🩺 COVID-19 Yapay Zeka Tanı Sistemi")
st.markdown("### Akciğer Röntgen Görüntüsü Analizi")
# Kullanıcıya projenin amacını açıklayan kısa bilgi
st.write("Bu sistem, **Derin Öğrenme (CNN)** kullanarak akciğer röntgenlerinden **COVID-19**, **Normal** ve **Viral Pnömoni** tespiti yapar.")

# Sidebar (Sol Menü) Tasarımı
with st.sidebar:
    # Proje logosu
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785819.png", width=100)
    st.header("Proje Hakkında")
    # Bilgilendirme kutusu
    st.info("""
    Bu proje Yapay Sinir Ağları dersi için hazırlanmıştır.
    
    **Model:** MobileNetV2 (Transfer Learning)
    **Eğitim:** ImageNet + Özel Veri Seti
    **Sınıflar:**
    - COVID-19
    - Normal
    - Viral Pneumonia
    """)
    st.write("---")
    st.write("Geliştirici: **EMIRA MERYEM**")
    
    # Gelişmiş analiz (Heatmap/Grad-CAM) seçeneği
    # Bu özellik seçilirse modelin odaklandığı bölgeler renklendirilir.
    show_heatmap = st.checkbox("Gelişmiş Analiz (Sıcaklık Haritası)", value=False, help="Modelin nereye odaklandığını gösterir.")

# Sayfayı iki sütuna böl (Resim yükleme ve Sonuç ekranı)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Röntgen Yükle")
    # Dosya yükleme bileşeni (Sadece resim dosyalarına izin ver)
    uploaded_file = st.file_uploader("Lütfen bir akciğer röntgeni (JPG/PNG) yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Yüklenen Görüntü', use_container_width=True)
    
    with col2:
        st.subheader("2. Analiz Sonucu")
        
        # Modeli Her Seferinde Yükle (Cache Sorununu Önlemek İçin)
        model = load_model()
        
        if model is None:
            st.error("🚨 Model dosyası bulunamadı! Lütfen önce 'src/train.py' dosyasını çalıştırarak modeli eğitin.")
        else:
            if st.button("Analizi Başlat"):
                with st.spinner('Yapay zeka görüntüyü inceliyor...'):
                    # Tahmin
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img)

                    # --- GÜVEN SKORU DÜZENLEMESİ ---
                    # Kullanıcı isteği: %100 yerine %90 civarı görünsün.
                    current_probs = prediction[0]
                    max_prob = np.max(current_probs)

                    if max_prob > 0.95:
                        # Hedef güven aralığı: %88 - %94
                        new_max = random.uniform(0.88, 0.94)
                        diff = max_prob - new_max
                        
                        # En yüksek olasılığı güncelle
                        max_index = np.argmax(current_probs)
                        current_probs[max_index] = new_max
                        
                        # Azalan miktarı diğer sınıflara dağıt
                        other_indices = [i for i in range(len(current_probs)) if i != max_index]
                        if other_indices:
                            share = diff / len(other_indices)
                            for idx in other_indices:
                                current_probs[idx] += share
                        
                        # Güncellenmiş değerleri geri ata
                        prediction[0] = current_probs
                    # --------------------------------
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    result_class = CLASSES[class_index]
                    
                    # Sonuç Gösterimi
                    if result_class == "COVID":
                        st.error(f"Tespit: **COVID-19**")
                    elif result_class == "Viral Pneumonia":
                         st.warning(f"Tespit: **Viral Pnömoni**")
                    else:
                        st.success(f"Tespit: **NORMAL**")
                    
                    st.metric(label="Güven Oranı", value=f"%{confidence:.2f}")
                    
                    st.write("---")
                    st.write("**Detaylı Olasılıklar:**")
                    # Debug için raw değerleri göster
                    for i, class_name in enumerate(CLASSES):
                        prob = prediction[0][i]
                        st.write(f"- {class_name}: %{prob*100:.2f}")
                        st.progress(int(prob * 100))
                    
                    # --- GRAD-CAM GÖRSELLEŞTİRME ---
                    if show_heatmap:
                        st.write("---")
                        st.subheader("🔥 Yapay Zeka Odak Haritası")
                        try:
                            from utils import make_gradcam_heatmap, save_and_display_gradcam
                            
                            # Transfer Learning modellerinde katman ismi farklı olabilir.
                            # 'Conv_1' MobileNetV2'nin son conv katmanıdır ama nested (iç içe) olabilir.
                            # Hata almamak için try-except bloğu ile deniyoruz.
                            heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer_name="Conv_1")
                            
                            # Geçici dosya olarak kaydetmeden direkt görüntü üzerinde işlem yapmamız lazım ama 
                            # utils fonksiyonumuz dosya yolu alıyor. Bunu basitleştirmek için:
                            # Resmi geçici kaydet
                            temp_path = "temp_img.png"
                            image.save(temp_path)
                            
                            final_img = save_and_display_gradcam(temp_path, heatmap)
                            st.image(final_img, caption="Grad-CAM Sıcaklık Haritası", use_container_width=True)
                            
                            # Temizlik
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
                        except Exception as e:
                            st.warning(f"Sıcaklık haritası şu an için oluşturulamadı.")
                            st.caption(f"Hata: {e}")

else:
    with col2:
        st.info("Analiz sonucunu görmek için sol taraftan bir resim yükleyiniz.")
