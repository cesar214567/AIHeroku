
FROM python:3.7-slim
ENV PYTHONUNBUFFERED=TRUE
RUN pip --no-cache-dir install pipenv
WORKDIR /app
COPY ["requirements.txt","./"]
RUN pip3 install -r requirements.txt
COPY ["*py","model.sav","./"]
EXPOSE 8080
ENTRYPOINT ["gunicorn","--bind","0.0.0.0:8080","main:app"]
