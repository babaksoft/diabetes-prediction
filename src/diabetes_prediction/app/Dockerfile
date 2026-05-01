# --- Builder Stage ---
FROM python:3.12.8-slim-bullseye AS builder

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml constraints.txt ./

RUN pip install --no-cache-dir --prefix=/install ".[app]"

# --- Final Production Image ---
FROM python:3.12.10-slim-bullseye

WORKDIR /app

COPY --from=builder /install /usr/local/

ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
