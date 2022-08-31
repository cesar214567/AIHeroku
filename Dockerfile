
FROM python:3.9.13-slim
ENV PYTHONUNBUFFERED=TRUE
RUN pip --no-cache-dir install pipenv
WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]
RUN pip3 install tensorflow
RUN pipenv install --deploy --system && rm -rf /root/.cache
COPY ["*py","model.sav","./"]
EXPOSE 8080
ENTRYPOINT ["gunicorn","--bind","0.0.0.0:8080","main:app"]
