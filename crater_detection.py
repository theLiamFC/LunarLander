
#model from Robotflow: https://universe.roboflow.com/coco-to-yolo-sybgr/moon-challenge/model/1

from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt
from lunar_render import tile #only used for example usage
import base64
import numpy as np


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="pNM8U9MqbgHzwRvULjL2"
)

# Path to your image file
# image_path = "lunar_images/tile1.jpg"

# # Read the image using OpenCV
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference


# img = cv2.cvtColor(tile.image, cv2.COLOR_RGB2GRAY)
# img = cv2.imread('/Users/ben/Desktop/School/Stanford/Quarter 3/AA 273/Final Project/LunarLander/lunar_images/tile1.jpg')
img_tile = np.repeat(tile[:, :, np.newaxis], 3, axis=2)
# img = cv2.imread('/Users/ben/Desktop/School/Stanford/Quarter 3/AA 273/Final Project/LunarLander/lunar_images/tile.jpg')
# print(img_tile)
# print(img)
# Encode to base64 string
# Encode image as JPEG in memory
# _, img_encoded = cv2.imencode('.jpg', img_tile)

# # Convert to byte stream
# img_bytes = img_encoded.tobytes()
# img_base64 = base64.b64encode(img_bytes).decode('utf-8')

# print(img)
# print('-------------')
# print(img_tile)

print('Before Inference')
result = CLIENT.infer(img_tile, model_id="moon-challenge/1")
print(result)
print('After Inference')
# image_rgb = img
# # Draw bounding boxes
# for prediction in result['predictions']:
#     x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
#     class_name = prediction['class']
#     confidence = prediction['confidence']

#     # Convert to top-left corner format
#     x1 = int(x - width / 2)
#     y1 = int(y - height / 2)
#     x2 = int(x + width / 2)
#     y2 = int(y + height / 2)

#     # Draw rectangle and label
#     cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
#     cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
# print(result['predictions'])

# # Show result
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.axis('off')
# plt.title("Object Detection Result")
# plt.show()

