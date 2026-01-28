# 🚀 QUICK START - Что сделать сейчас

## Локально (тест)

```bash
# 1. Обнови зависимости
pip install --upgrade -r requirements.txt

# 2. .env уже обновлен - проверь что есть FIREBASE_CRED_JSON
cat .env | grep FIREBASE

# 3. Запусти тест
python main.py

# Должно быть в логах:
# ✅ Firebase storage initialized
# ✅ Bot is running...
```

---

## На Railway (деплой)

```
1. GitHub → Push коммит с изменениями
2. Railway → Deployments (автоматически)
3. Railway → Settings → Variables → Добавь:
   FIREBASE_CRED_JSON={весь JSON из файла}
4. Railway → Deployments → View logs → Проверь ✅
```

---

## Файлы которые изменились

```diff
✏️  config.py
✏️  memory.py
✏️  models.py
✏️  gemini_analyzer.py
✏️  responder.py
✏️  requirements.txt
✏️  .env
✏️  DEPLOY.md
+ ✨ CHANGES.md
+ ✨ RAILWAY_SETUP.md
+ ✨ EXPLANATION.md
```

---

## Что изменилось (TL;DR)

| Проблема | Решение |
|----------|---------|
| Firebase не работает на Railway | FIREBASE_CRED_JSON вместо файла |
| Google API устарел | google-generativeai → google-genai |
| Стикер ID жестко закодирован | STICKER_PACK_ID в .env |

---

## Проверка

✅ Локальная версия работает?
```bash
python main.py
# ищи в логах "Bot is running..."
```

✅ GitHub синхронизирован?
```bash
git status
# должно быть всё committed
```

✅ Railway переменные добавлены?
```
Settings → Variables → FIREBASE_CRED_JSON есть?
```

Если всё ✅ - готово! 🎉
