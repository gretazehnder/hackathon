FROM python:3.11-slim
WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi uvicorn scikit-learn joblib numpy

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
