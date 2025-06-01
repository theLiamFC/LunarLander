#model from Robotflow: https://universe.roboflow.com/coco-to-yolo-sybgr/moon-challenge/model/1

from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
import matplotlib.pyplot as plt
from api_secrets import RF_API_KEY

class CraterDetector:
    def __init__(self):
        """
        Initiates the CraterDetector class.

        Parameters
        ----------
        """
        self.__result = None
        self.__predictions = None
        
        self.CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=RF_API_KEY
        )
    
    def __infer__(self, img):
        """
        Runs the model on the image to detect craters. 
        
        The infer method returns a JSON dictionary with one key, 'predictions', which contains a list of dictionaries.
        Each dictionary contains information about the detected object. 
        """
        self.__result = self.CLIENT.infer(img, model_id="moon-challenge/1")
        self.__result = self.__result['predictions']
        
    def __prepare_results__(self, img):
        """
        Prepares the results after the infer method has been called. The JSON dictionary will be transformed into a numpy array where 
        each row is a unique detection. THE NUMPY ARRAY THAT STORES THE INFORMATION FOR THE PREDICTIONS CONTAINS THE FOLLOWING INFORMATION MATCHING THE JSON:
        - x: x-coordinate at the center of the box
        - y: y-coordinate at the center of the box
        - width: width of the bounding box
        - height: height of the bounding box
        - confidence: confidence of the detection
        """
        self.__infer__(img) #run inference on the image
        rows = len(self.__result)
        predictions = np.zeros((rows, 5))
        
        for i, values in enumerate(self.__result):
            x = values['x']
            y = values['y']
            w = values['width']
            h = values['height']
            conf = values['confidence']
            predictions[i,:] = np.array([x,y,w,h,conf])
        
        self.__predictions = predictions
    
    def detect_craters(self, tile):
        """
        Getter Method: this method is how a user will be able to access the crater predictions.
        """
        self.__prepare_results__(np.repeat(tile.image[:, :, np.newaxis], 3, axis=2))
        return self.__predictions
        
    def view_craters(self, tile):
        """
        Getter Method: enables users to visualize the craters that were detected
        """
        rgb_image = cv2.cvtColor(tile.image, cv2.COLOR_BGR2RGB)
        predictions = self.detect_craters(tile)
        print('working with predictions')
        for prediction in predictions:
            x,y,width,height = prediction[:4]
            conf = prediction[4]
            
            # Convert to top-left corner format
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
        
        
            # Draw rectangle and label
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(rgb_image, f"Crater with Conf = {conf:.2f}", (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
        

        # Show result
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title("Object Detection Result")
        plt.show()
    
    def estimate_crater_radius(self, img):
        """
        Performs a simple average of the width and the height to get the crater diameter. 
        Then returns the radius with the formula r = d/2

        Returns:
            estimated radius of each crater
        """
        predictions = self.detect_craters(img)
        diameter = (predictions[:,2] + predictions[:,3]) / 2
        return diameter/2
    
    def gather_image_points(self, img):
        """
        Gathers points on the crater in the image that will be used to estimate.
        Args:
            img (_type_): _description_
        """
        predictions = self.detect_craters(img)
        r = self.estimate_crater_radius(img)
        
        m = predictions.shape[0]
        n = 5 #first point will be original center followed by top, right, bottom, left points
        add_points = np.zeros((n, 2*m))
        
        original_points = predictions[:,:2].reshape(1,-1) #gather all the x,y positions 
        
        
    
        
            
            
        
        
        
    
# example usage
if __name__ == "__main__":
    from lunar_render import LunarRender
    moon = LunarRender('../WAC_ROI')
    tile = moon.render_m(x=-2000, y=-50000, alt=50000)
    
    detector = CraterDetector()
    detector.view_craters(tile)
    print(detector.estimate_crater_radius(tile))