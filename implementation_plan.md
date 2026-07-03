# Implementation Plan

## [Overview]
Веб-приложение для распознавания лиц с использованием FastAPI, которое позволяет загружать фотографии известных людей в базу, сканировать директорию с целевыми фото, распознавать известных людей и автоматически обнаруживать новых (неизвестных) людей с помощью кластеризации DBSCAN.

## [Architecture Summary]
Приложение работает по принципу «Веб-интерфейс + AI Pipeline»:
1. Пользователь через браузер загружает фото известных людей с указанием имени
2. Приложение извлекает вектора лиц (ArcFace embeddings) и сохраняет в Qdrant
3. Пользователь выбирает директорию для сканирования
4. Приложение сканирует все фото, сравнивает с базой известных, группирует неизвестных
5. Результат отображается в браузере: известные + найденные новые люди

## [Tech Stack]
- **Язык**: Python 3.10+
- **Веб-фреймворк**: FastAPI
- **UI**: HTML/TailwindCSS (Jinja2 шаблоны)
- **ML Library**: DeepFace
- **Модель**: ArcFace
- **Детектор**: RetinaFace
- **Векторная БД**: Qdrant (локальный/w Docker)
- **Кластеризация**: scikit-learn (DBSCAN)

## [Types]

### FaceResult (Результат распознавания)
```python
class FaceResult:
    face_id: str              # Уникальный ID лица
    image_path: str           # Путь к изображению
    bbox: dict                # {'x', 'y', 'w', 'h'}
    embedding: List[float]    # 512-мерный вектор ArcFace
    is_known: bool           # Распознан как известный
    name: str | None         # Имя если известный
    confidence: float | None # similarity score (0-1)
```

### PersonCluster (Кластер неизвестного человека)
```python
class PersonCluster:
    cluster_id: int           # ID кластера
    face_count: int           # Количество фото этого лица
    image_paths: List[str]   # Список путей к фото
    sample_embedding: List[float] # Средний вектор кластера
```

### ScanResult (Результат сканирования)
```python
class ScanResult:
    total_images: int        # Всего обработано фото
    known_faces: List[FaceResult]   # Найденные известные
    unknown_clusters: List[PersonCluster] # Новые люди
```

## [API Endpoints]

### POST /api/known/add
- Input: form-data с файлами и именем person_name
- Logic: Извлечь вектора лиц → сохранить в Qdrant коллекцию "known_faces"
- Response: {"status": "ok", "faces_count": N}

### POST /api/scan
- Input: form-data с путём к директории
- Logic: 
  1. Сканировать все фото в директории
  2. Для каждого лица: поиск в Qdrant (known_faces)
  3. Если не найден → добавить в список неизвестных
  4. Применить DBSCAN к неизвестным
- Response: ScanResult JSON

### GET /api/results
- Вернуть последний езультат сканирования

### GET /api/known/list
- Список всех известных людей в базе

## [Files]

### Структура проекта
```
FaceRecognize/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Конфигурация
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic модели
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_recognizer.py  # DeepFace обёртка
│   │   ├── vector_db.py        # Qdrant клиент
│   │   └── clustering.py       # DBSCAN логика
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── api.py             # API эндпоинты
│   │   └── pages.py           # HTML страницы
│   └── templates/
│       ├── base.html
│       ├── index.html
│       └── results.html
├── data/
│   └── known_faces/         # Директория для загрузки известных
├── requirements.txt
├── docker-compose.yml       # Qdrant + app
├── .env.example
└── README.md
```

### Новые файлы для создания

| Файл | Назначение |
|------|-------------|
| app/main.py | Главный модуль FastAPI |
| app/config.py | Настройки приложения |
| app/models/schemas.py | Pydantic модели данных |
| app/services/face_recognizer.py | Сервис распознавания DeepFace |
| app/services/vector_db.py | Qdrant клиент |
| app/services/clustering.py | DBSCAN кластеризация |
| app/routers/api.py | API эндпоинты |
| app/routers/pages.py | HTML роуты |
| app/templates/*.html | Jinja2 шаблоны |
| requirements.txt | Зависимости |
| docker-compose.yml | Запуск Qdrant |

## [Functions]

### face_recognizer.py
```python
def extract_faces(image_path: str) -> List[FaceResult]:
    """Извлечь все лица из изображения"""
    
def extract_faces_from_directory(dir_path: str) -> List[FaceResult]:
    """Извлечь все лица из всех фото директории"""
    
def get_embedding(image_path: str) -> List[float]:
    """Получить embedding одного лица"""
```

### vector_db.py
```python
def init_qdrant() -> QdrantClient:
    """Инициализировать подключение к Qdrant"""
    
def add_known_face(name: str, embedding: List[float], image_path: str) -> str:
    """Добавить известное лицо в базу"""
    
def search_known(embedding: List[float], threshold: float = 0.6) -> Optional[MatchResult]:
    """Поиск наиболее похожего лица в базе"""
    
def get_all_known() -> List[dict]:
    """Получить список всех известных"""
```

### clustering.py
```python
def cluster_unknown_faces(embeddings: List[List[float]], 
                          images: List[str],
                          eps: float = 0.4, 
                          min_samples: int = 3) -> List[PersonCluster]:
    """Кластеризация неизвестных лиц DBSCAN"""
```

## [Classes]

### FaceRecognizerService (services/face_recognizer.py)
- Отвечает за извлечение лиц из изображений
- Использует DeepFace.represent() с ArcFace
- Методы: extract_faces(), extract_faces_from_directory()

### VectorDBService (services/vector_db.py)
- Обертка над Qdrant клиентом
- Управляет коллекцией known_faces
- Методы: add_known_face(), search_known(), get_all_known()

### ClusteringService (services/clustering.py)
- Инкапсулирует логику DBSCAN
- Группирует неизвестные лица
- Методы: cluster_unknown_faces()

### FaceRecognizer (app/main.py)
- Главный класс приложения FastAPI
- Объединяет все сервисы
- Содержит эндпоинты API

## [Dependencies]

### Python пакеты (requirements.txt)
```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
jinja2>=3.1.0
deepface>=0.7.0
qdrant-client>=1.6.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

### Docker (docker-compose.yml)
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
```

## [Testing]

### Тестирование компонентов
1. **Unit тесты**: Тестирование отдельных функций
2. **Integration тесты**: Тестирование API эндпоинтов
3. **E2E тесты**: Полный цикл через браузер

### Валидация
- Тест загрузки известных лиц
- Тест распознавания известного лица
- Тест кластеризации неизвестных (2 фото одного человека = 1 кластер)
- Тест случая "неизвестный" (1 фото = шум/-1)

### Тестовые данные
- known_faces/ - тестовые фото известных людей для CI

## [Implementation Order]

### Этап 1: Базовая инфраструктура
1. Создать requirements.txt
2. Создать config.py с конфигурацией
3. Настроить docker-compose.yml для Qdrant
4. Создать базовые модели Pydantic (schemas.py)

### Этап 2: Core сервисы
5. Реализовать face_recognizer.py - извлечение лиц через DeepFace
6. Реализовать vector_db.py - Qdrant клиент
7. Реализовать clustering.py - DBSCAN кластеризация

### Этап 3: API слой
8. Создать main.py с базовой конфигурацией FastAPI
9. Реализовать api.py - эндпоинты /api/known/add, /api/scan
10. Создать простые HTML шаблоны

### Этап 4: UI и интеграция
11. Создать templates/index.html - главная страница
12. Создать templates/results.html - страница результатов
13. Реализовать pages.py - роуты для HTML
14. Добавить статические файлы (CSS)

### Этап 5: Тестирование и документация
15. Написать базовые тесты
16. Создать README.md с инструкцией по запуску
17. Финальное тестирование full pipeline

## [Configuration]

### Параметры (config.py)
```python
# Qdrant настройки
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# DeepFace настройки
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

# Поиск
KNOWN_THRESHOLD = 0.6  # Cosine similarity порог
CLUSTERING_EPS = 0.4   # DBSCAN радиус
CLUSTERING_MIN_SAMPLES = 3  # Минимум фото для кластера

# Paths
KNOWN_FACES_DIR = "./data/known_faces"
```

### Переменные окружения (.env)
```
QDRANT_HOST=localhost
QDRANT_PORT=6333
KNOWN_THRESHOLD=0.6