FROM python:3.7.5

RUN pip install fastapi
RUN pip install uvicorn[standard]
RUN pip install azure-cosmos
RUN pip install requests
RUN pip install azure-storage-blob
RUN pip install datasets

COPY ./ ./

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


