# Membuat Environment Conda berdasarkan `environtment.yml`

1. Buka PowerShell baru.
2. Pindah ke direktori proyek dan buat environment:

   ```
   cd {direktori_proyek}
   conda env create -f environtment.yml
   ```
3. Aktifkan environment:

   ```
   conda activate final-mlops
   ```

---

# Deployment Model TFX ke Railway Menggunakan Docker

## 1. Menyiapkan Dockerfile

Pastikan file `Dockerfile` berada di **root directory** repo dan ditulis seperti berikut:

```dockerfile
FROM tensorflow/serving:latest
COPY ./mrickyr-pipeline/serving_model/mrickyr-pipeline /models/diabetes-prediction

ENV MODEL_NAME=diabetes-prediction
ENV MODEL_BASE_PATH=/models
ENV PORT=8501

# Script entrypoint agar mengikuti PORT dari Railway
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \\
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \\
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
    && chmod +x /usr/bin/tf_serving_entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
```

---

## 2. Uji Dockerfile secara Lokal

Sebelum deploy ke Railway, build dan test Docker secara lokal:

```bash
docker build -t diabetes-serving .
docker run -p 8501:8501 diabetes-serving
```

Jika model berjalan dengan benar, endpoint lokal dapat diakses via:

```
http://localhost:8501/v1/models/diabetes-prediction
```

---

## 3. Push Proyek ke GitHub

Pastikan seluruh file, termasuk `Dockerfile` dan folder `serving_model`, sudah masuk ke repo.

```bash
git init
git remote add origin <URL_REPO_GITHUB>
git add .
git commit -m "Notes"
git branch -M main
git push -u origin main
```

---

## 4. Deploy ke Railway

1. Masuk ke [https://railway.app](https://railway.app)
2. Klik **New Project → Deploy from GitHub Repo**
3. Pilih repository
4. Railway otomatis mendeteksi Dockerfile dan mulai melakukan build

Jika build berhasil, status akan menjadi **Success**.

---

## 5. Mengaktifkan Domain Publik

Masuk ke tab **Deployments → Networking** lalu klik **Generate Domain**.

---

## 6. Melakukan Prediksi via REST API

Model TFX menggunakan `tf.train.Example` sebagai input, sehingga request harus dalam bentuk base64 dari serialized Example.

Contoh Python yang berhasil:

```python
import requests
import tensorflow as tf
import base64

url = "https://<DOMAIN_RAILWAY>/v1/models/diabetes-prediction:predict"

data = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
    "Outcome": 1,
}

def int_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

def float_feature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

feature_dict = {
    "Pregnancies": int_feature(data["Pregnancies"]),
    "Glucose": int_feature(data["Glucose"]),
    "BloodPressure": int_feature(data["BloodPressure"]),
    "SkinThickness": int_feature(data["SkinThickness"]),
    "Insulin": int_feature(data["Insulin"]),
    "BMI": float_feature(data["BMI"]),
    "DiabetesPedigreeFunction": float_feature(data["DiabetesPedigreeFunction"]),
    "Age": int_feature(data["Age"]),
    "Outcome": int_feature(data["Outcome"]),
}

example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
raw_bytes = example.SerializeToString()
b64 = base64.b64encode(raw_bytes).decode("utf-8")

payload = {
    "instances": [
        {"b64": b64}
    ]
}

res = requests.post(url, json=payload)
print("STATUS:", res.status_code)
print("RESPONSE:", res.json())
```

---


