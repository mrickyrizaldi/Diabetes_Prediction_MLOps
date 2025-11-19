# Gunakan image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Salin folder model hasil push
COPY ./mrickyr-pipeline/serving_model/mrickyr-pipeline /models/diabetes-prediction

# Nama model untuk TensorFlow Serving
ENV MODEL_NAME=diabetes-prediction

# Base path semua model
ENV MODEL_BASE_PATH=/models

# Default port
ENV PORT=8501

# Buat entrypoint script supaya TF Serving pakai PORT dari environment
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
    && chmod +x /usr/bin/tf_serving_entrypoint.sh

# Expose port REST API
EXPOSE 8501

# Jalankan TF Serving via script
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
