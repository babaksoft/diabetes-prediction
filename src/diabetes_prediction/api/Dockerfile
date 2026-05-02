FROM python:3.12.8-slim-bullseye AS builder

RUN apt-get update --fix-missing

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml constraints.txt ./

RUN pip install --upgrade pip build

COPY src/ ./src/

RUN python -m build --wheel --outdir /dist

FROM python:3.12.8-slim-bullseye

RUN apt-get update --fix-missing

WORKDIR /app

COPY --from=builder /dist /dist
COPY constraints.txt /constraints.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -c /constraints.txt /dist/*.whl && \
    WHEEL=$(ls /dist/*.whl) && \
    pip install --no-cache-dir -c /constraints.txt "${WHEEL}[api]"

EXPOSE 8000

CMD ["uvicorn", "diabetes_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
