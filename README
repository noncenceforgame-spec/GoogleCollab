Шаг 1: Сборка Docker образа

Откройте терминал в директории, где находятся ваши файлы (Dockerfile, streamlit_app.py, requirements.txt, final_gradient_boosting_model.pkl, scaler.pkl, start.sh) и выполните следующую команду:

docker build -t credit-status-app .
docker build: Команда для сборки образа Docker.
-t credit-status-app: Присваивает имя (credit-status-app) и тег (по умолчанию latest) вашему образу.
.: Указывает, что Dockerfile находится в текущей директории.
Шаг 2: Запуск Docker контейнера

После успешной сборки образа вы можете запустить контейнер:

docker run -p 8501:8501 --name my-credit-app credit-status-app
docker run: Команда для запуска контейнера из образа.
-p 8501:8501: Пробрасывает порт 8501 из контейнера на порт 8501 вашей хост-машины. Streamlit по умолчанию использует порт 8501.
--name my-credit-app: Присваивает имя my-credit-app вашему контейнеру (для удобства управления).
credit-status-app: Имя образа, который мы хотим запустить.
Ваше Streamlit приложение будет доступно в браузере по адресу http://localhost:8501 (или IP-адресу вашей Docker-машины, если вы используете удаленный Docker).

Запуск контейнера в фоновом режиме (detached mode):

Чтобы запустить контейнер в фоновом режиме и продолжить использовать терминал, добавьте флаг -d:

docker run -d -p 8501:8501 --name my-credit-app credit-status-app
Остановка и удаление контейнера:

Остановить контейнер:

docker stop my-credit-app
Удалить контейнер:

docker rm my-credit-app
Удаление образа (если он больше не нужен):

docker rmi credit-status-app
