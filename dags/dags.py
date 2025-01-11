from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.sensors.filesystem import FileSensor

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='training',
    default_args=default_args,
    schedule_interval="* * * * *",
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    file_change = FileSensor(
        task_id='monitor_change_in_params',
        filepath="/Users/muntakim/Developer/Project/Personal_Dev/LINKEDIN_DEMO/PROJECT_1/params.yaml",
        poke_interval=5,  # check every 60 seconds
        timeout=60 # timeout after 10 minutes
    )
    trigger = BashOperator(
        task_id='dvc_stater',
        bash_command="cd /Users/muntakim/Developer/Project/Personal_Dev/LINKEDIN_DEMO/PROJECT_1/ && /Users/muntakim/Developer/Project/Personal_Dev/LINKEDIN_DEMO/PROJECT_1/.venv/bin/dvc repro"
    )
    file_change >> trigger