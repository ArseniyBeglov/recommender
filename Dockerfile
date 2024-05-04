# Используем официальный образ Python в качестве базового образа
FROM python:3.10-slim-buster

# Создаем директорию /app внутри контейнера
WORKDIR /recommender

# Устанавливаем gcc
RUN apt-get update && apt-get install -y gcc

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
CMD ["main.py"]
