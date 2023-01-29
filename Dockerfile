FROM python:3.8
WORKDIR /app 
COPY requirements.txt /app/
RUN pip install -r requirements.txt 


COPY src/app.py /app/
COPY src/slregression_model.pkl /app/
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 