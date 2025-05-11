FROM python:3.11-bookworm

WORKDIR /

RUN pip install pipenv

COPY Pipfile* ./

RUN pipenv install --system

RUN pipenv sync

COPY . .

EXPOSE 3000

CMD [ "pipenv", "run", "python", "application.py", "production"]