FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# opencv-python 运行所需的系统库
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

# 离线模型挂载目录（宿主机 ./creator/models -> 容器 /app/creator/models）
VOLUME ["/app/creator/models"]

EXPOSE 8081

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1", "--limit-concurrency", "9999", "--backlog", "1024"]
