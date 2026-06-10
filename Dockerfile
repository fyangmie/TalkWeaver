FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY packages.txt /tmp/packages.txt
RUN apt-get update \
    && xargs -r apt-get install -y --no-install-recommends < /tmp/packages.txt \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "webapp/streamlit_app.py", "--server.address=0.0.0.0"]
