from locust import task, TaskSet
import os
from os.path import isfile, join
from os import listdir
from locust_tasks.commons.local_data_store import get_encoded_image
from loguru import logger

# Get current path
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

# Pre-define all path for different resolution images and pdfs
IMG_RESOURCES = {
    "low-res-img": f"{CUR_PATH}/resources/load-test-low-res-image_computer.jpg",
    "mid-res-img": f"{CUR_PATH}/resources/load-test-mid-res-image_street_view.jpg",
    "high-res-img": f"{CUR_PATH}/resources/load-test-high-res-image_lavatory.jpg",
    "ultra-res-img": f"{CUR_PATH}/resources/load-test-ultrahigh-res-image_office.jpg",
    "low-res-doc": f"{CUR_PATH}/resources/load-test-detect-text-image_receipt.jpg",
    "low-res-pdf": f"{CUR_PATH}/resources/load-test-detect-text-image_one_page.pdf",
    "high-res-pdf": f"{CUR_PATH}/resources/load-test-detect-text-image_essay.pdf",
    "ultra-res-doc": f"{CUR_PATH}/resources/load-test-detect-text-image_big_small.jpg",
    "random-images": f"{CUR_PATH}/resources/images",
    "random-documents": f"{CUR_PATH}/resources/documents",
}

TESTING_IMG = {
    "ocr": f"{CUR_PATH}/resources/load-test-detect-text-image_receipt.jpg",
    "dc": f"{CUR_PATH}/resources/book-page.jpg",
    "ld": f"{CUR_PATH}/resources/book-page.jpg",
    "ic": f"{CUR_PATH}/resources/load-test-low-res-image_computer.jpg",
    "od": f"{CUR_PATH}/resources/load-test-low-res-image_computer.jpg",
}


class MultiDataImageModelTask(TaskSet):
    """Load test models running in OKE. Only support IC/OD and OCR models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body = None

    def on_start(self):
        """
        Initializing signer and request body for http clients.
        """
        self.model_name = os.environ.get("MODEL_NAME")
        self.body = self._define_request_body()
        logger.info(self.client.base_url)
        self.len_body = len(self.body)
        self.idx = 0

    @task(1)
    def predict(self):
        """
        Sending request to prediction endpoint, and log prediction latency.
        """
        rsp = self.client.post("/predict", json=self.body[self.idx%self.len_body])
        self.idx += 1

    def _define_request_body(self):
        """
        Define request body for model prediction, only supports IC/OD and OCR models.
        """
        request_bodies = []
        model_name = os.environ.get("MODEL_NAME")
        image_path = None
        file_type = "image"
        features = None
        if "ic" in model_name:
            image_path = TESTING_IMG.get("ic", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "IMAGE_CLASSIFICATION",
                    "maxResults": 5,
                }
            ]
        elif "dc" in model_name:
            image_path = TESTING_IMG.get("dc", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "DOCUMENT_CLASSIFICATION",
                    "maxResults": 5,
                }
            ]
            file_type = "document"
        elif "od" in model_name:
            image_path = TESTING_IMG.get("od", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "OBJECT_DETECTION",
                    "maxResults": 5,
                }
            ]
        elif "ld" in model_name:
            image_path = TESTING_IMG.get("ld", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "LANGUAGE_CLASSIFICATION",
                    "maxResults": 5,
                }
            ]
            file_type = "document"
        elif "kv" in model_name:
            image_path = TESTING_IMG.get("ocr", IMG_RESOURCES["low-res-img"])
            features = [
                {"featureType": "KEY_VALUE_DETECTION"},
            ]
            file_type = "document"
        elif "ocr" in model_name:
            image_path = TESTING_IMG.get("ocr", IMG_RESOURCES["low-res-img"])
            features = [
                {"featureType": "TEXT_DETECTION"},
            ]
            file_type = "document"
        elif "td" in model_name:
            image_path = TESTING_IMG.get("ocr", IMG_RESOURCES["low-res-img"])
            features = [
                {"featureType": "TABLE_DETECTION"},
            ]
            file_type = "document"
        else:
            raise Exception(
                f"Unrecognized model name: {model_name}. Unable to generate requests.")

        image_path = f"{CUR_PATH}/resources/images"
        images = [f for f in listdir(
            image_path) if isfile(join(image_path, f))]

        for image in images:
            _, image_byte = get_encoded_image(
                image_name=f"{CUR_PATH}/resources/images/{image}"
            )
            body = {
                "features": features,
                file_type: {"data": image_byte.decode()},
                "maxResults": 5
            }
            request_bodies.append(body)
        return request_bodies


class ImageModelTasks(TaskSet):
    """Load test models running in OKE. Only support IC/OD and OCR models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body = None

    def on_start(self):
        """
        Initializing signer and request body for http clients.
        """
        self.body = self._define_request_body()
        logger.info(self.client.base_url)

    @task(1)
    def predict(self):
        """
        Sending request to prediction endpoint, and log prediction latency.
        """
        rsp = self.client.post("/predict", json=self.body)

    def _define_request_body(self):
        """
        Define request body for model prediction, only supports IC/OD and OCR models.
        """
        model_name = os.environ.get("MODEL_NAME")
        image_path = None
        file_type = "image"
        features = None
        if "ic" in model_name:
            image_path = TESTING_IMG.get("ic", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "IMAGE_CLASSIFICATION",
                    "maxResults": 5,
                }
            ]
        elif "dc" in model_name:
            image_path = TESTING_IMG.get("dc", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "DOCUMENT_CLASSIFICATION",
                    "maxResults": 5,
                }
            ]
            file_type = "document"
        elif "od" in model_name:
            image_path = TESTING_IMG.get("od", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "OBJECT_DETECTION",
                    "maxResults": 5,
                }
            ]
        elif "ld" in model_name:
            image_path = TESTING_IMG.get("ld", IMG_RESOURCES["low-res-img"])
            features = [
                {
                    "featureType": "LANGUAGE_CLASSIFICATION",
                    "maxResults": 5,
                }
            ]
            file_type = "document"
        elif "kv" in model_name:
            image_path = TESTING_IMG.get("ocr", IMG_RESOURCES["low-res-img"])
            features = [
                {"featureType": "KEY_VALUE_DETECTION"},
            ]
            file_type = "document"
        elif "ocr" in model_name:
            image_path = TESTING_IMG.get("ocr", IMG_RESOURCES["low-res-img"])
            features = [
                {"featureType": "TEXT_DETECTION"},
            ]
            file_type = "document"
        elif "td" in model_name:
            image_path = TESTING_IMG.get("ocr", IMG_RESOURCES["low-res-img"])
            features = [
                {"featureType": "TABLE_DETECTION"},
            ]
            file_type = "document"
        else:
            raise Exception(f"Unrecognized model name: {model_name}. Unable to generate requests.")

        _, image_byte = get_encoded_image(
            image_name=image_path
        )
        image_name = image_path.split("/")[-1]
        logger.info(
            f"Testing model: {model_name}. Testing image: {image_name}")

        body = {
            "features": features,
            file_type: {"data": image_byte.decode()},
        }
        return body
