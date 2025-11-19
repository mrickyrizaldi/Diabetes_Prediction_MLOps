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
git add .
git commit -m "Notes"
git remote add origin https://github.com/mrickyrizaldi/Diabetes_Prediction_MLOps.git
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

## 6. Cek Status Serving

```python
import requests
from pprint import PrettyPrinter

url = "https://diabetespredictionmlops-production.up.railway.app/v1/models/diabetes-prediction"

res = requests.get(url)

pp = PrettyPrinter()
pp.pprint(res.json())
print("\nSTATUS:", res.status_code)
```

---

# Monitoring Model Serving dengan Prometheus & Grafana
Bagian ini mendokumentasikan langkah yang saya lakukan untuk mengaktifkan monitoring pada model **diabetes-prediction** yang dideploy menggunakan **TensorFlow Serving di Railway**, kemudian dimonitor dengan **Prometheus** dan divisualisasikan menggunakan **Grafana**.

---

## Menyiapkan Konfigurasi Monitoring untuk TensorFlow Serving

1. **Membuat berkas `prometheus.config`** di dalam folder `config/` yang kemudian disalin ke dalam image Docker model serving.

   isi `config/prometheus.config`:

   ```
   prometheus_config {
     enable: true,
     path: "/monitoring/prometheus/metrics"
   }
   ```

   Penjelasan singkat:

   * `enable: true` → mengaktifkan endpoint monitoring Prometheus di TensorFlow Serving.
   * `path` → path HTTP yang akan diekspos oleh TensorFlow Serving untuk metrik.

2. **Memodifikasi Dockerfile model serving** supaya:

   * `prometheus.config` ikut disalin ke dalam container.
   * TensorFlow Serving dijalankan dengan flag `--monitoring_config_file`.

   berikut isi `Dockerfile` terbarunya:

   ```Dockerfile
    FROM tensorflow/serving:latest
    COPY ./mrickyr-pipeline/serving_model/mrickyr-pipeline /models/diabetes-prediction

    # Salin config untuk Prometheus monitoring
    COPY ./config /model_config

    ENV MODEL_NAME=diabetes-prediction
    ENV MODEL_BASE_PATH=/models

    # File konfigurasi monitoring
    ENV MONITORING_CONFIG=/model_config/prometheus.config

    ENV PORT=8501

    # Entry point script: tambah flag --monitoring_config_file
    RUN echo '#!/bin/bash \n\n\
    env \n\
    tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
    --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    --monitoring_config_file=${MONITORING_CONFIG} \
    "$@"' > /usr/bin/tf_serving_entrypoint.sh \
        && chmod +x /usr/bin/tf_serving_entrypoint.sh

    EXPOSE 8501
    ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
   ```

3. **Redeploy image ke Railway** (build & push lewat GitHub → Railway akan rebuild image dan menjalankan container baru). Setelah itu, endpoint metrics dapat diakses melalui:

```

https://<nama-app-railway>.up.railway.app/monitoring/prometheus/metrics

```

Pada project ini endpoint-nya berupa:

```

[https://diabetespredictionmlops-production.up.railway.app/monitoring/prometheus/metrics](https://diabetespredictionmlops-production.up.railway.app/monitoring/prometheus/metrics)

```

---

## Menyiapkan Prometheus untuk Scraping Metrik dari Railway

1. **Membuat folder `monitoring/`** yang berisi konfigurasi dan Dockerfile untuk Prometheus:

Struktur sederhana:

```

monitoring/
├─ Dockerfile
└─ prometheus.yml

````

2. **Isi `monitoring/prometheus.yml`**

File ini berisi konfigurasi global Prometheus serta job untuk scraping TensorFlow Serving di Railway.

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: "tf-serving-railway-monitor"

scrape_configs:
  - job_name: "tf-serving-railway"
    scrape_interval: 5s
    metrics_path: /monitoring/prometheus/metrics
    scheme: https
    static_configs:
      - targets: ['diabetespredictionmlops-production.up.railway.app']
````

Penjelasan singkat:

* `scrape_interval: 5s` → Prometheus menarik metrik dari TF Serving setiap 5 detik.
* `metrics_path` → harus sama dengan path pada `prometheus.config` di TensorFlow Serving.
* `scheme: https` → karena endpoint Railway sudah menggunakan HTTPS.
* `targets` → domain Railway tanpa `https://` dan tanpa path.

3. **Isi `monitoring/Dockerfile`** untuk Prometheus:

   ```Dockerfile
   FROM prom/prometheus:latest

   COPY prometheus.yml /etc/prometheus/prometheus.yml
   ```

4. **Build dan jalankan container Prometheus** secara lokal:

   ```bash
   docker build -t diabetes-monitoring ./monitoring
   docker run -p 9090:9090 --name diabetes-prometheus diabetes-monitoring
   ```

5. **Verifikasi target di Prometheus**

   * Buka: `http://localhost:9090`
   * Masuk ke menu **Status → Targets**
   * Pastikan job `tf-serving-railway` berstatus **UP**.

   Jika status UP, berarti Prometheus berhasil melakukan scraping metrik dari TensorFlow Serving di Railway.

---

### Eksplorasi dan Query Metrik di Prometheus

Setelah Prometheus terhubung, saya melakukan eksplorasi metrik melalui tab **Graph** di `http://localhost:9090` dan endpoint metrics Railway. Beberapa metrik penting yang muncul dari TensorFlow Serving antara lain:

* `:tensorflow:serving:request_count` → jumlah request yang diterima model.

Contoh query yang digunakan di Prometheus:

* **Total request yang diterima model**

  ```promql
  :tensorflow:serving:request_count{model_name="diabetes-prediction"}
  ```

---

## Menyiapkan Grafana dan Menyambungkan ke Prometheus

1. **Menjalankan Grafana dengan Docker**

   ```bash
   docker run -d -p 3000:3000 --name diabetes-grafana grafana/grafana
   ```

   Setelah container berjalan, dashboard Grafana bisa diakses melalui:

   ```
   http://localhost:3000
   ```

   Login default:

   * Username: `admin`
   * Password: `admin` (akan diminta untuk diganti saat login pertama).

2. **Menambahkan Prometheus sebagai Data Source di Grafana**

   * Buka menu **Configuration → Data sources**.
   * Klik **Add data source** dan pilih **Prometheus**.
   * Isi URL sesuai lokasi Prometheus:
     * Jika Prometheus berjalan langsung di host: `http://localhost:9090`
   * Klik **Save & Test** dan pastikan statusnya **Data source is working**.

3. **Membuat Dashboard Monitoring**

   * Buka menu **Create → Dashboard → Add new panel**.
   * Pilih data source: **Prometheus**.
   * Isi query PromQL sesuai metrik yang ingin dimonitor.
   * Simpan dashboard dengan nama `Diabetes_Prediction-monitoring`.

---

## Panel dan Metrik yang Dimonitor di Grafana

Berikut beberapa panel yang saya buat di Grafana beserta alasan pemilihan metriknya.


---

### Monitoring API / Serving Metrics

Pada bagian ini, saya memanfaatkan metrik bawaan TensorFlow Serving untuk memantau performa endpoint `diabetes-prediction`. Semua metrik berbasis pada keluarga metrik `:tensorflow:serving:*`.

#### 1. Total API Requests

**Tujuan:** mengetahui berapa banyak request yang sudah diterima model.

**PromQL:**

```promql
sum(:tensorflow:serving:request_count{model_name="diabetes-prediction"})
```

**Alasan:**

* Memberikan gambaran total penggunaan model sejak service berjalan.
* Cocok ditampilkan sebagai panel **Stat** atau **Time series** sederhana.

---

#### 2. API Request by Status Code

**Tujuan:** melihat distribusi request berdasarkan status (misalnya `OK`, `ERROR`).

**PromQL:**

```promql
sum by (status) (
  :tensorflow:serving:request_count{model_name="diabetes-prediction"}
)
```

**Alasan:**

* Memudahkan identifikasi apakah ada peningkatan jumlah request yang gagal.
* Meskipun saat ini status yang muncul masih `OK`, panel ini tetap berguna sebagai dasar monitoring error di masa depan.

---

#### 3. API Throughput (Requests per Second / RPS)

**Tujuan:** memantau laju request per detik (traffic API).

**PromQL:**

```promql
rate(:tensorflow:serving:request_count{model_name="diabetes-prediction"}[1m])
```

(Alternatif lebih halus:)

```promql
rate(:tensorflow:serving:request_count{model_name="diabetes-prediction"}[5m])
```

**Alasan:**

* Menunjukkan seberapa padat traffic yang masuk ke model.
* Membantu melihat pola penggunaan (jam sepi/padat) dan potensi bottleneck.

---

#### 4. API Errors

**Tujuan:** menghitung banyaknya request yang tidak berstatus `OK` dan persentase error.

**PromQL – jumlah error per detik:**

```promql
rate(
  :tensorflow:serving:request_count{
    model_name="diabetes-prediction",
    status!="OK"
  }[5m]
)
```

**PromQL – error rate (persentase):**

```promql
rate(:tensorflow:serving:request_count{model_name="diabetes-prediction",status!="OK"}[5m])
/
rate(:tensorflow:serving:request_count{model_name="diabetes-prediction"}[5m])
```

**Alasan:**

* Panel ini dapat digunakan untuk mendeteksi gangguan pada sistem (misalnya meningkatnya error 5xx/4xx).
* Saat ini nilainya cenderung 0 (semua request sukses), namun tetap penting sebagai indikator reliabilitas.

---

#### 5. Average API Latency

**Tujuan:** memantau rata-rata waktu respon model (latency) untuk endpoint REST `predict`.

TensorFlow Serving menyediakan:

* `:tensorflow:serving:request_latency_sum{...}`
* `:tensorflow:serving:request_latency_count{...}`

**PromQL:**

```promql
(
  :tensorflow:serving:request_latency_sum{
    model_name="diabetes-prediction",
    API="predict",
    entrypoint="REST"
  }
/
  :tensorflow:serving:request_latency_count{
    model_name="diabetes-prediction",
    API="predict",
    entrypoint="REST"
  }
) / 1000
```

**Alasan:**

* Mengukur performa rata-rata API dari sisi pengguna (end-to-end latency).
* Dibagi 1000 untuk mengubah satuan dari microseconds menjadi milliseconds sehingga lebih mudah dibaca di dashboard.

---

#### 6. API Latency Worst Case (p95 / p99)

**Tujuan:** melihat latency "terburuk" (tail latency) dengan menggunakan quantile dari histogram latency.

Metrik yang digunakan: `:tensorflow:serving:request_latency_bucket{...}`.

**PromQL – p95:**

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(
      :tensorflow:serving:request_latency_bucket{
        model_name="diabetes-prediction",
        API="predict",
        entrypoint="REST"
      }[5m]
    )
  )
)
```

**PromQL – p99 (opsional):**

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(
      :tensorflow:serving:request_latency_bucket{
        model_name="diabetes-prediction",
        API="predict",
        entrypoint="REST"
      }[5m]
    )
  )
)
```

**Alasan:**

* Rata-rata latency sering tidak cukup untuk menggambarkan pengalaman pengguna terburuk.
* p95/p99 membantu mendeteksi masalah performa yang hanya muncul pada sebagian kecil request.

---

### Traffic Anomaly Detection (Low Traffic & Spike)

Sebagai tambahan, PromQL juga dapat digunakan untuk mendefinisikan rule deteksi anomali traffic (misalnya dalam Alertmanager atau alert di Grafana).

#### 1. Low Traffic Detector

**Tujuan:** mendeteksi kondisi traffic yang terlalu rendah (misalnya jika tidak ada request sama sekali).

**PromQL (contoh):**

```promql
rate(:tensorflow:serving:request_count{model_name="diabetes-prediction"}[5m]) < 0.01
```

**Alasan:**

* Mengindikasikan bahwa dalam 5 menit terakhir hampir tidak ada request yang masuk.
* Dapat digunakan untuk memberi peringatan jika sistem tidak lagi digunakan atau terjadi gangguan pada client.

---

#### 2. Request Spike Detector

**Tujuan:** mendeteksi lonjakan request yang tidak biasa (spike traffic).

**PromQL (contoh):**

```promql
rate(:tensorflow:serving:request_count{model_name="diabetes-prediction"}[1m])
>
(
  rate(:tensorflow:serving:request_count{model_name="diabetes-prediction"}[15m]) * 3
)
```

**Alasan:**

* Membandingkan RPS 1 menit terakhir dengan rata-rata 15 menit sebelumnya.
* Jika nilai saat ini 3x lebih tinggi dari rata-rata, bisa dianggap sebagai spike dan berpotensi membutuhkan scaling atau investigasi lebih lanjut.

---




