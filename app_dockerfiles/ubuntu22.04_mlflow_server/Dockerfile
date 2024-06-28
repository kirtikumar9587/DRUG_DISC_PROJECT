FROM python:3.10.14-bookworm

RUN pip3 install mlflow==2.12.2

RUN pip3 install mysql-connector-python

RUN pip3 install mysqlclient

EXPOSE 5000

RUN mkdir -p /mlruns

RUN chmod 777 /mlruns

CMD mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mysql://mlflow_user:mlflow@localhost:3306/db_mlflow --default-artifact-root /mlruns
