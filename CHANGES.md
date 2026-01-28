# 📋 Что изменилось - Полное объяснение

## 1️⃣ Firebase: Переход с файла на переменную окружения

### Проблема:
- Когда бот запускается в контейнере (Docker/Railway), файл `firebase-adminsdk-fbsvc-04d73b3287.json` недоступен
- Образуется ошибка: `[Errno 2] No such file or directory`

### Решение:
Вместо хранения JSON в файле, используем переменную окружения `FIREBASE_CRED_JSON`

### Где изменилось:

#### **config.py:**
```python
# ❌ БЫЛО (требовал файл):
firebase_cred_path=_get_required_env("FIREBASE_CRED_PATH")

# ✅ СТАЛО (поддерживает JSON строку):
firebase_cred_path=_get_firebase_credentials() or ""
```

Новая функция `_get_firebase_credentials()`:
- Сначала проверяет `FIREBASE_CRED_JSON` (JSON строка для Production)
- Если нет, ищет `FIREBASE_CRED_PATH` (файл для Development)
- Если файл существует, читает его содержимое

#### **memory.py:**
```python
# ❌ БЫЛО:
cred = credentials.Certificate(cred_path)  # Ожидает только файл

# ✅ СТАЛО:
if cred_path.strip().startswith('{'):
    cred_dict = json.loads(cred_path)  # Парсим JSON строку
    cred = credentials.Certificate(cred_dict)
else:
    cred = credentials.Certificate(cred_path)  # Файл как раньше
```

#### **.env:**
```
# ❌ БЫЛО:
FIREBASE_CRED_PATH=deep-seek-chat-manager-firebase-adminsdk-fbsvc-04d73b3287.json

# ✅ СТАЛО:
FIREBASE_CRED_JSON={"type":"service_account",...}
```

### Как использовать:

**Локально:**
- Либо используй FIREBASE_CRED_JSON (JSON в одну строку)
- Либо FIREBASE_CRED_PATH с путем к файлу (если файл существует)

**На Railway:**
- Создай переменную `FIREBASE_CRED_JSON`
- Скопируй весь содержимое JSON файла в одну строку
- JSON автоматически распарсится при запуске

---

## 2️⃣ Deprecated Google Generative AI → New google-genai

### Проблема:
```
FutureWarning: All support for the google.generativeai package has ended.
Please switch to the google.genai package as soon as possible.
```

### Решение:
Обновили на новый пакет `google-genai`

### Где изменилось:

#### **requirements.txt:**
```
# ❌ БЫЛО:
google-generativeai>=0.3.0

# ✅ СТАЛО:
google-genai>=0.1.0
```

#### **gemini_analyzer.py:**

Импорт:
```python
# ❌ БЫЛО:
import google.generativeai as genai

# ✅ СТАЛО:
import google.genai as genai
```

Инициализация:
```python
# ❌ БЫЛО:
genai.configure(api_key=api_key)
self._model = genai.GenerativeModel('gemini-2.0-flash')

# ✅ СТАЛО:
self._client = genai.Client(api_key=api_key)
self._model = 'gemini-2.0-flash'
```

Вызов API:
```python
# ❌ БЫЛО:
response = self._model.generate_content(prompt)

# ✅ СТАЛО:
response = self._client.models.generate_content(
    model=self._model,
    contents=prompt
)
```

---

## 3️⃣ Стикерпак из конфига

### Что изменилось:

#### **models.py:**
```python
# Добавлено поле:
sticker_pack_id: str = "userpack7845974bystickrubot"
```

#### **.env:**
```
# Добавлено:
STICKER_PACK_ID=userpack7845974bystickrubot
```

#### **config.py:**
```python
# Добавлено в load_config():
sticker_pack_id=os.getenv("STICKER_PACK_ID", "userpack7845974bystickrubot"),
```

#### **responder.py:**
```python
# В __init__:
if config.sticker_pack_id:
    self._stickers.load_set(config.sticker_pack_id)
```

### Преимущества:
- Стикерпак загружается автоматически при инициализации
- Легко менять ID через переменную окружения
- Не нужно искать в коде

---

## 🚀 На что влияют эти изменения:

| Компонент | До | После |
|-----------|-----|-------|
| **Локальный запуск** | Нужен файл `firebase-adminsdk-...json` | Опционально - можно использовать .env |
| **Railway деплой** | Нужно загружать файл (проблемы с доступом) | Просто добавить переменную `FIREBASE_CRED_JSON` |
| **Gemini API** | Warning при каждом запуске | Чистые логи, актуальный пакет |
| **Стикеры** | Жестко закодированы в коде | Настраиваются через .env |

---

## ✅ Чек-лист перед деплоем на Railway:

- [ ] Установлены новые зависимости: `pip install google-genai`
- [ ] Переменная `FIREBASE_CRED_JSON` добавлена на Railway (вся строка JSON)
- [ ] `STICKER_PACK_ID` установлен (если использовать кастомный)
- [ ] `.env` файл с `FIREBASE_CRED_JSON` на локальной машине
- [ ] `firebase-adminsdk-*.json` файл НЕ в GitHub
- [ ] Тестовый запуск локально без ошибок

---

## 📝 Возможные проблемы и решения:

### Problem 1: "Invalid JSON in FIREBASE_CRED_JSON"
**Решение:** Убедись что JSON в одну строку и без посторонних символов:
```python
# Неправильно (с переносами):
{"type":"service_account",
 "project_id":"..."}

# Правильно (одна строка):
{"type":"service_account","project_id":"..."}
```

### Problem 2: "Gemini API call failed"
**Решение:** Проверь что используется `google-genai`, не `google-generativeai`:
```bash
pip list | grep google
# Должно быть: google-genai 0.1.0 или выше
```

### Problem 3: "Sticker pack invalid"
**Решение:** Проверь ID в Railway и локально:
```bash
# Локально проверить тест
python -c "from config import get_config; print(get_config().sticker_pack_id)"
```
