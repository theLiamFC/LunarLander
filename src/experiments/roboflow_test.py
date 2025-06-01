import requests
from api_secrets import RF_API_KEY
from lunar_render import LunarRender

from inference_sdk import InferenceHTTPClient

moon = LunarRender('WAC_ROI',debug=False)
tile = moon.render(u=40000, v=0, alt=100000)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="BK7uj0ymwTtsVtZYIjEk"
)

try:
    result = CLIENT.infer(tile.image, model_id="coco/3")
    print("Public model works - issue with your models")
except Exception as e:
    print(f"All models failing: {e}")

result = CLIENT.infer(tile.image, model_id="moon-challenge/1")