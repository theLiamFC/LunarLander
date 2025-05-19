
#model from Robotflow: https://universe.roboflow.com/coco-to-yolo-sybgr/moon-challenge/model/1

from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt
from LunarRender import tile #only used for example usage

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
result = CLIENT.infer(tile.image, model_id="moon-challenge/1")
image_rgb = tile.image
# Draw bounding boxes
for prediction in result['predictions']:
    x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
    class_name = prediction['class']
    confidence = prediction['confidence']

    # Convert to top-left corner format
    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    # Draw rectangle and label
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    cv2.putText(image_rgb, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
print(result['predictions'])

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Object Detection Result")
plt.show()

