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
    pip install --no-cache-dir -c /constraints.txt "${WHEEL}[app]"

EXPOSE 8501

ENTRYPOINT ["sh", "-c", "streamlit run $(python -c 'import diabetes_prediction.app.streamlit_app as m; print(m.__file__)') --server.port=8501 --server.address=0.0.0.0"]
