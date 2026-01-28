# 🚀 Railway: Пошаговая инструкция по деплою

## Шаг 1: Подготовка Firebase JSON

1. Открой файл `deep-seek-chat-manager-firebase-adminsdk-fbsvc-04d73b3287.json` 
2. Скопируй **весь** его содержимое
3. Это будет значение для переменной `FIREBASE_CRED_JSON`

**⚠️ ВАЖНО:** JSON должен быть в **одну строку** (без переносов)

Пример как должно выглядеть (первые символы):
```
{"type":"service_account","project_id":"deep-seek-chat-manager",...}
```

## Шаг 2: На Railway добавь переменные

1. Откройте свой проект на railway.app
2. Перейди: **Settings** → **Variables** (или **Environment**)
3. Добавь следующие переменные:

```
TELEGRAM_TOKEN=8290363081:AAFwpAeNtgB1smFYDpTi5FZIGtN1TST-nco
DEEPSEEK_API_KEY=sk-f2c4d91e31714ae0a0af82fef2933fd1
GIPHY_API_KEY=EBWSyiu3IlrVJM2cmWJAnTyityGrNbzl
GEMINI_API_KEY=AIzaSyD2cig_pSw84sFKygK8rHpHRg9SYCFSMfk
BOT_NAME=DeepSeek
STICKER_PACK_ID=userpack7845974bystickrubot
FIREBASE_CRED_JSON={копируй весь JSON отсюда}
```

### Как добавить FIREBASE_CRED_JSON правильно:

1. Скопируй содержимое файла `deep-seek-chat-manager-firebase-adminsdk-fbsvc-04d73b3287.json`
2. На Railway в поле **Value** вставь весь JSON
3. Должно получиться как-то так:
   ```
   {"type":"service_account","project_id":"deep-seek-chat-manager","private_key_id":"04d73b3287..."}
   ```

## Шаг 3: Гит и коммит

1. Убедись что `.env` файл в `.gitignore` (не выгружается в GitHub)
2. Убедись что `firebase-adminsdk-*.json` в `.gitignore`
3. Сделай коммит:
   ```bash
   git add config.py memory.py models.py gemini_analyzer.py requirements.txt DEPLOY.md CHANGES.md
   git commit -m "fix: Update Firebase to use env variable instead of file, migrate to google-genai, add sticker_pack_id config"
   git push origin main
   ```

## Шаг 4: Railway автоматически деплойит

- Railway видит push в GitHub
- Автоматически скачивает код
- Использует переменные из Settings
- Запускает бота

Проверь логи в Railway → **Deployment** → **View logs**

---

## ✅ Что произойдёт:

### До изменений (ошибка):
```
[err] Firebase not available
[err] Failed to initialize Firebase: No such file or directory: 'firebase-*.json'
```

### После изменений (работает):
```
[inf] Firebase storage initialized
[inf] Bot initialized with all components
[inf] Bot is running... Press Ctrl+C to stop
```

---

## 🔧 Местные переменные (.env)

Для локальной разработки в `.env`:

```
TELEGRAM_TOKEN=8290363081:AAFwpAeNtgB1smFYDpTi5FZIGtN1TST-nco
DEEPSEEK_API_KEY=sk-f2c4d91e31714ae0a0af82fef2933fd1
GIPHY_API_KEY=EBWSyiu3IlrVJM2cmWJAnTyityGrNbzl
GEMINI_API_KEY=AIzaSyD2cig_pSw84sFKygK8rHpHRg9SYCFSMfk
BOT_NAME=DeepSeek
STICKER_PACK_ID=userpack7845974bystickrubot

# Вариант 1: JSON в одну строку
FIREBASE_CRED_JSON={"type":"service_account","project_id":"..."}

# ИЛИ Вариант 2: Путь к файлу (для локальной разработки)
# FIREBASE_CRED_PATH=deep-seek-chat-manager-firebase-adminsdk-fbsvc-04d73b3287.json
```

---

## 🆘 Если что-то не работает:

### Провери переменные на Railway:
```
Railway → Project → Settings → Variables
```
Убедись что все нужные переменные добавлены

### Проверь логи:
```
Railway → Project → Deployments → последний деплой → View logs
```

### Если Firebase сообщает ошибку:
Может быть в FIREBASE_CRED_JSON неправильное форматирование JSON

Правильный способ скопировать:
1. Открой файл в текстовом редакторе (VS Code)
2. Выдели ВСЕ содержимое Ctrl+A
3. Скопируй Ctrl+C
4. На Railway вставь Ctrl+V в поле Value

---

## 📊 Summary

| Что | Локально | Railway |
|-----|----------|---------|
| **Firebase** | Может быть файл или JSON строка | ✅ JSON строка только |
| **Переменные** | В .env файле | В Settings → Variables |
| **.env** | ❌ В .gitignore (не грузить) | - (не используется) |
| **firebase-*.json** | ✅ Может быть локально | ❌ НЕ грузить в GitHub |

---

## 🎯 Когда готово:

1. ✅ Переменные добавлены на Railway
2. ✅ Код закоммичен и запушен
3. ✅ Railway деплойилась (посмотри логи)
4. ✅ Бот в логах пишет "Bot is running..."
5. ✅ Отправь сообщение боту в Telegram

Готово! 🚀
