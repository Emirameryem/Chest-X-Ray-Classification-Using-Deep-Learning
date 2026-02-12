import os
import requests

# Gerçek verilerden birkaç örnek (GitHub/Public kaynaklardan)
REAL_SAMPLES = {
    "COVID": [
        "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/01E392EE-69F9-4E33-BFCE-E5F9756529C7.jpeg",
        "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/03BF7561-A9BA-4C3C-B8A0-D3E585F73F3C.jpeg",
        "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/0a7faa2a.jpg",
        "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/0b45780d.jpg"
    ],
    "NORMAL": [
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/normal/normal_100.png",
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/normal/normal_101.png",
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/normal/normal_102.png",
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/normal/normal_103.png"
    ],
    "Viral Pneumonia": [
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/viral/viral_10.png",
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/viral/viral_11.png",
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/viral/viral_12.png",
        "https://raw.githubusercontent.com/jinyu121/COVID-19-Detection/master/dataset/viral/viral_13.png"
    ]
}

DATA_DIR = os.path.join(os.getcwd(), "data")

def download_file(url, folder, filename):
    try:
        response = requests.get(url, timeout=10)
        path = os.path.join(DATA_DIR, folder, filename)
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"İndirildi: {folder}/{filename}")
        return True
    except Exception as e:
        print(f"Hata ({filename}): {e}")
        return False

def setup_real_demo_data():
    print("Gerçek demo verileri indiriliyor...")
    
    # Mevcut dummy verileri temizle (İsteğe bağlı, karışmasın diye)
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if "dummy" in f:
                os.remove(os.path.join(root, f))
    
    for category, urls in REAL_SAMPLES.items():
        folder_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        for i, url in enumerate(urls):
            fname = f"real_demo_{i}.png"
            if "jpeg" in url: fname = f"real_demo_{i}.jpg"
            
            download_file(url, category, fname)
            
    print("İndirme tamamlandı! Şimdi 'python src/train.py' çalıştırabilirsin.")

if __name__ == "__main__":
    setup_real_demo_data()
