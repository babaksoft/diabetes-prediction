# --- Builder Stage ---
FROM python:3.12.10-slim-bullseye AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Final Production Image ---
FROM python:3.12.10-slim-bullseye

WORKDIR /app

COPY --from=builder /install /usr/local/

ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
