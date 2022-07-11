from locust import HttpUser
from locust_tasks.vs_model_tasks import ImageModelTasks, MultiDataImageModelTask
import os

SERVICE_NAME=os.environ.get("SERVICE_NAME")
SERVICE_NAMESPACE=os.environ.get("SERVICE_NAMESPACE")
HOST = f"http://{SERVICE_NAME}.{SERVICE_NAMESPACE}:8080" if SERVICE_NAME and SERVICE_NAMESPACE else "http://localhost:8080"

class ImageModel(HttpUser):
    host = HOST
    tasks = [MultiDataImageModelTask]
