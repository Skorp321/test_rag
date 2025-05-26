# requirements.txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-mistralai>=0.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
beautifulsoup4>=4.12.0
unstructured>=0.10.0
python-dotenv>=1.0.0

# Инструкции по установке и запуску

## 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

## 2. Получение API ключа Mistral AI
1. Зарегистрируйтесь на https://console.mistral.ai/
2. Получите API ключ в разделе API Keys
3. Введите ключ в интерфейсе приложения

## 3. Подготовка HTML документа
Подготовьте HTML файл (doc.html) с контентом, по которому будет отвечать агент.

## 4. Запуск приложения
```bash
streamlit run app.py
```

## 5. Использование
1. Введите Mistral API ключ в боковую панель
2. Загрузите HTML документ через интерфейс
3. Нажмите "Обработать документ"
4. Задавайте вопросы в чате

## Возможные проблемы и решения

### Ошибка с FAISS
Если возникают проблемы с FAISS, попробуйте:
```bash
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir
```

### Ошибка с sentence-transformers
Первый запуск может занять время из-за загрузки модели эмбеддингов.

### Проблемы с unstructured
Если возникают ошибки с unstructured, установите дополнительные зависимости:
```bash
pip install unstructured[local-inference]
```

## Альтернативная версия без unstructured
Если у вас проблемы с библиотекой unstructured, в коде используется BeautifulSoup для парсинга HTML.