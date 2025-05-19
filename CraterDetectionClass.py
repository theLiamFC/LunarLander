#model from Robotflow: https://universe.roboflow.com/coco-to-yolo-sybgr/moon-challenge/model/1

from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
import matplotlib.pyplot as plt
from LunarRender import tile #only used for example usage




class CraterDetector:
    def __init__(self, tile):
        """
        Initiates the CraterDetector class.

        Parameters
        ----------
        tile: Tile from LunarRender Class File. The object contains an image, information about it's global and window size in m
        """
        self.tile = tile
        self.__result = None
        self.__predictions = None
        
        self.CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="pNM8U9MqbgHzwRvULjL2"
        )
        print('Initialized')
    
    def __infer__(self):
        """
        Runs the model on the image to detect craters. 
        
        The infer method returns a JSON dictionary with one key, 'predictions', which contains a list of dictionaries.
        Each dictionary contains information about the detected object. 
        """
        print('Infered')
        self.__result = self.CLIENT.infer(self.tile.image, model_id="moon-challenge/1")
        print(self.__result)
        self.__result = self.__result['predictions']
        print('Exiting Infer')
        
    def __prepare_results__(self):
        """
        Prepares the results after the infer method has been called. The JSON dictionary will be transformed into a numpy array where 
        each row is a unique detection. THE NUMPY ARRAY THAT STORES THE INFORMATION FOR THE PREDICTIONS CONTAINS THE FOLLOWING INFORMATION MATCHING THE JSON:
        - x: x-coordinate at the center of the box
        - y: y-coordinate at the center of the box
        - width: width of the bounding box
        - height: height of the bounding box
        - confidence: confidence of the detection
        """
        self.__infer__() #run inference on the image
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
        print('prepare')
    
    def detect_craters(self):
        """
        Getter Method: this method is how a user will be able to access the crater predictions.
        """
        print('Detect Crater Method')
        if self.__predictions is not None:
            return self.__predictions
        else:
            self.__prepare_results__()
            return self.__predictions
        
    def view_craters(self):
        """
        Getter Method: enables users to visualize the craters that were detected
        """
        predictions = self.detect_craters()
        img = self.tile.image
        for prediction in predictions:
            x,y,width,height = predictions[:4]
            conf = predictions[5]
            
            # Convert to top-left corner format
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
        
        
            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(img, f"Crater with Conf = {conf:.2f}", (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
        

        # Show result
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Object Detection Result")
        plt.show()
    
    def estimate_crater_radius(self):
        """
        Performs a simple average of the width and the height to get the crater diameter. 
        Then returns the radius with the formula r = d/2

        Returns:
            estimated radius of each crater
        """
        predictions = self.detect_craters()
        diameter = (predictions[:,2] + predictions[:,3]) / 2
        return diameter/2
    
#example usage
print(CraterDetector(tile).estimate_crater_radius())