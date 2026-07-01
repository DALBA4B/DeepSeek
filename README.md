# DeepSeek Telegram Bot 🤖

Групповой Telegram бот на Python, который использует DeepSeek AI для интеллектуальных ответов. Бот читает сообщения, анализирует контекст и отвечает естественным образом, используя различные форматы ответов (текст, реакции, гифки, стикеры).

## Особенности

- ✅ **AI-powered responses** — использует DeepSeek API для генерации ответов
- ✅ **Multiple response formats** — текст, реакции emoji, гифки, стикеры
- ✅ **Smart memory system** — краткосрочная (RAM) и долгосрочная (Firebase) память
- ✅ **Natural conversation** — 10% случайный ответ, реагирует на упоминания и вопросы
- ✅ **Secure credentials** — все секреты хранятся в `.env` файле
- ✅ **Modular architecture** — чистый, типизированный код с dependency injection
- ✅ **Async HTTP** — неблокирующие запросы к Giphy API
- ✅ **Graceful shutdown** — корректное завершение работы

## Требования

- Python 3.8+
- Telegram Bot Account (получи token у [@BotFather](https://t.me/botfather))
- DeepSeek API Key (от https://api.deepseek.com)
- Giphy API Key (от https://developers.giphy.com)
- Firebase Firestore (с credentials JSON файлом)

## Быстрый старт

### 1. Клонировать/Загрузить проект

```bash
cd DeepSeek
```

### 2. Создать виртуальное окружение (рекомендуется)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# или
source venv/bin/activate  # Linux/Mac
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Настроить переменные окружения

Скопируй `.env.example` в `.env` и заполни реальные значения:

```bash
cp .env.example .env
```

Отредактируй `.env` файл со своими ключами (см. комментарии в файле).

### 5. Запустить бота

```bash
python main.py
```

## Структура проекта

```
.
├── .env                    # Переменные окружения (НЕ коммитить!)
├── .env.example            # Шаблон для .env с документацией
├── .gitignore              # Git ignore (защищает .env)
├── config.py               # Загрузка и валидация конфигурации
├── models.py               # Модели данных (dataclasses)
├── prompts.py              # Системные промпты для DeepSeek
├── utils.py                # Общие хелперы (timezone-aware время)
├── main.py                 # Точка входа, класс DeepSeekBot
├── memory.py               # Двухуровневая память (RAM + Firebase)
├── brain.py                # V2: классификация (0-3) + LightRAG + генерация
├── conversation_analyzer.py # Fast-классификатор: grade 0-3 + needs_memory + rag_query
├── rag_client.py           # Async-клиент LightRAG (retrieve/insert/clear/health)
├── rag_ingestor.py         # Ночной pipeline: сбор → группировка по времени → insert
├── responder.py            # Отправка разных типов ответов (текст/реакция/гиф/стикер)
├── night_analyzator.py     # Планировщик + RagIngestTask (ночная индексация)
├── graph_memory.py         # ⚠️ Устаревший граф знаний (отключён, оставлен для справки)
├── deepseek_analyzer.py    # ⚠️ Устаревший ночной анализатор (отключён, оставлен для справки)
├── requirements.txt        # Зависимости Python
├── railway.json            # Конфиг деплоя на Railway
├── render.yaml             # Конфиг деплоя на Render
└── README.md               # Этот файл
```

## Архитектура (Фаза B — LightRAG)

### Поток обработки сообщения (V2)

```
Telegram Message
    ↓
DeepSeekBot.handle_message()
    ├── Фильтрация (боты, chat_id)
    ├── memory.add_message()  ← сохраняем (вкл. reply_to_message)
    ↓
[Шаг 1] Brain.analyze_and_respond()
    ├── ConversationAnalyzer.classify()   ← 1 вызов fast DeepSeek
    │     → { grade: 0-3, needs_memory, rag_query }
    ├── grade == 0?  → молчим
    ├── needs_memory? → RagClient.retrieve(rag_query)  ← LightRAG (только факты)
    └── generate_response()                ← main DeepSeek (grade → формат/токены)
    ↓
Responder.send_response()
    └── ResponseParser → TEXT / REACT / GIPHY / STICKER
    ↓
Telegram Response
```

### Градация ответов (grade 0-3)

| grade | Что делает бот |
|-------|----------------|
| 0 | Молчит (флуд, не к нему, неинтересно) |
| 1 | Короткая реакция (стикер/гифка/смайл/1-3 слова) |
| 2 | Обычный содержательный ответ |
| 3 | Развёрнутый ответ с упором на факты из LightRAG |

### Ночной pipeline (LightRAG)

```
Планировщик (run_hour из .env)
    ↓
RagIngestTask.run()
    └── RagIngestor.ingest()
          ├── Сбор сообщений (Firebase → fallback daily_log)
          ├── Группировка по времени (блоки 10-15 мин, reply_to вшит в блок)
          └── rag_client.insert(block) по одному → LightRAG делает extraction + embeddings
    ↓
Отчёт в чат + обновление курсора (идемпотентность)
```

### Команды

| Команда | Действие |
|---------|----------|
| `/daily_log` | Сообщения за сегодня (для отладки) |
| `/ragstats` | Статус LightRAG + последняя индексация |
| `/ragnow` | Вручную запустить индексацию за 24ч |
| `/profile <имя>` | Что бот знает о человеке (из LightRAG) |
| `/ragclean confirm` | ⚠️ Полностью очистить базу знаний |

## Безопасность

### Переменные окружения

⚠️ **НИКОГДА не коммитьте `.env` файл с реальными секретами!**

`.gitignore` уже настроен для исключения `.env` файлов.

### Защита учетных данных

- ✅ Все токены и ключи в `.env`
- ✅ Firebase credentials в отдельном JSON файле
- ✅ `.gitignore` защищает конфиденциальные файлы
- ✅ `config.py` валидирует наличие всех переменных

## Тестирование

Автоматические тесты пока не подключены (в планах на V2). Для проверки
запусти бота локально и понаблюдай за логами (см. «Быстрый старт»):

```bash
python main.py
```

## Кастомизация

### Изменение личности бота

В `prompts.py` отредактируй функцию `get_system_prompt()`:

```python
def get_system_prompt(bot_name: str, available_stickers: List[str]) -> str:
    return f"""Ты {bot_name} - [твое описание персоны]
    ...
    """
```

### Изменение имени и вариаций упоминаний

Имя бота берётся из `BOT_NAME` в `.env` и автоматически подставляется в персону.
Вариации для детекта упоминаний строятся из `BOT_NAME` функцией
`get_name_variations()` в `prompts.py` (точное написание, слитно и через дефис),
так что менять код не нужно — достаточно поменять `BOT_NAME`.

### Добавление стикеров

```python
from responder import StickerManager

stickers = StickerManager()
stickers.add_sticker("excited", "CAACAgIAAxkBAAE...")
```

### Настройка параметров

В `.env` файле:

```env
SHORT_MEMORY_LIMIT=50        # Больше памяти
CONTEXT_MESSAGES_COUNT=30    # Больше контекста
DEEPSEEK_MAX_TOKENS=200      # Длиннее ответы
DEEPSEEK_TEMPERATURE=0.5     # Менее креативно
```

## Логирование

```
2026-01-24 11:35:07 - __main__ - INFO - Bot is running...
2026-01-24 11:35:21 - memory - INFO - Message added - John: какая погода?
2026-01-24 11:35:21 - brain - INFO - Should respond: question mark detected
2026-01-24 11:35:22 - brain - INFO - Generated response: похоже будет дождик
```

Уровень логирования: `LOG_LEVEL` в `.env` (DEBUG, INFO, WARNING, ERROR)

## Ошибки и решения

### "TELEGRAM_TOKEN not found"
1. Проверь что `.env` файл существует
2. Проверь что `TELEGRAM_TOKEN=...` заполнен
3. Перезапусти бота

### "Cannot reach Telegram API"
- Проверь интернет соединение
- Проверь правильность токена
- Убедись что бот добавлен в группу

### "DeepSeek API Error"
- Проверь баланс DeepSeek аккаунта
- Проверь правильность API key
- Проверь интернет соединение

### "Firebase initialization failed"
- Проверь путь к credentials JSON
- Проверь что файл существует и валиден
- Бот продолжит работу без долгосрочной памяти

## Лицензия

MIT License — используй свободно в своих проектах

## Контрибьютинг

Улучшения и баг-репорты приветствуются!

---

**Начни разговор с ботом! 🚀**
