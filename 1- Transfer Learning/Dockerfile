FROM tensorflow/tensorflow:2.18.0-gpu as base
WORKDIR /
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt update \
    && apt install -y --no-install-recommends libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data

FROM base AS final
WORKDIR /app
COPY ./main.py ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
VOLUME /app/data
ENTRYPOINT [ "python", "./main.py" ]
