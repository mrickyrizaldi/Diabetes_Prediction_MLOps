# Gunakan image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Salin model hasil training TFX
COPY ./mrickyr-pipeline/serving_model/mrickyr-pipeline /models/diabetes-prediction

# Salin config untuk Prometheus monitoring
COPY ./config /model_config

# Nama model & base path
ENV MODEL_NAME=diabetes-prediction
ENV MODEL_BASE_PATH=/models

# File konfigurasi monitoring
ENV MONITORING_CONFIG=/model_config/prometheus.config

# Port REST API (di-overwrite Railway via env PORT)
ENV PORT=8501

# Entry point script: tambah flag --monitoring_config_file
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
    && chmod +x /usr/bin/tf_serving_entrypoint.sh

# Expose port REST API
EXPOSE 8501

# Jalankan TF Serving via script
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
