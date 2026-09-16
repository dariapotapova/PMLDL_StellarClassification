# References: https://hub.docker.com/_/python
FROM python:3.11.0-slim-bullseye@sha256:b5ee5360b357dcd0ab2e1cc5e22fa589b6734bbd55776df6b1700994febf1527

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt requirements.lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.1 \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && python -m pip install --no-cache-dir -r requirements.lock.txt \
    && python -m pip check

COPY . .
RUN chmod +x scripts/reproduce.sh

CMD ["./scripts/reproduce.sh"]
