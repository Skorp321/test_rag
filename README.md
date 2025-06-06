# Запуск Document QA Agent через Docker

Это приложение представляет собой Streamlit веб-интерфейс для работы с LangChain агентом, который отвечает на вопросы по содержанию HTML документов, используя Mistral AI.

## Быстрый запуск

1. **Клонируйте репозиторий и перейдите в директорию:**
   ```bash
   git clone git@github.com:Skorp321/test_rag.git
   cd test_rag
   ```

2. **Создайте файл .env с вашим API ключом:**
   ```bash
   echo "MISTRAL_API_KEY=your_mistral_api_key_here" > .env
   ```

3. **Запустите приложение:**
   ```bash
   docker compose up --build
   ```

4. **Откройте браузер и перейдите по адресу:**
   ```
   http://localhost:8501
   ```

## Использование приложения

1. **Загрузка документа:**
   - В боковой панели выберите способ загрузки файла
   - Загрузите свой HTML файл или используйте файл по умолчанию (doc.html)
   - Нажмите "Обработать документ"

2. **Задавание вопросов:**
   - После успешной обработки документа появится чат
   - Задавайте вопросы по содержанию документа
   - Агент будет отвечать на основе загруженного контента