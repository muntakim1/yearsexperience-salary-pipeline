mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5500 &

airflow db init & 
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
    
airflow scheduler &
airflow webserver &
