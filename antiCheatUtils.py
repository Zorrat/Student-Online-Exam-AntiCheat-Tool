####### Utility and model loading #############
import cv2
import colorsys
import random
import numpy as np
import tensorflow as tf
import face_recognition as faceRec
from keras.layers import Activation
from keras.models import Model
from keras.utils import get_custom_objects

### Custom Class Inheritance ######

class Mish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'



def mysoftplus(x):

    mask_min = tf.cast((x<-20.0),tf.float32)
    ymin = mask_min*tf.math.exp(x)

    mask_max = tf.cast((x>20.0),tf.float32)
    ymax = mask_max*x
    
    mask= tf.cast((abs(x)<=20.0),tf.float32)
    y = mask*tf.math.log(tf.math.exp(x) + 1.0)
    
    return(ymin+ymax+y)    
        

def mish(x):
    return (x* tf.math.tanh(mysoftplus(x)))


get_custom_objects().update({'mish': Mish(mish)})

print('Loading.....')
# Load the model
from keras.models import load_model,Model
yolo_model = load_model("models/yolo/yolov4.h5")


############# Helper Util Funcs #################

def read_labels(labels_path):
    with open(labels_path) as f:
        labels = f.readlines()
    labels = [c.strip() for c in labels]
    return labels
# Load the labels
labels = read_labels("models/yolo/coco_classes.txt")
#Manually Taken index for Cellphones
cellphone_idx = 67

# load and prepare an image
def load_image_pixels(image, shape):
    
    # load the CV image to get its shape
    width, height,_ = image.shape
    # load the image with the required size
    image = cv2.resize(image,shape)
    # convert to numpy array
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    #Normalize Image
    image /= 255.0
    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)   
    return image,width,height

######### Bounding Box Class to store bounding box info for easier access #########
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
 
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
 
        return self.label
 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
 
        return self.score



######### Helper Functions: Lots of them ################
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
 
def decode_netout(netout, anchors, obj_thresh, net_h, net_w, anchors_nb, scales_x_y):
    grid_h, grid_w = netout.shape[:2]  
    nb_box = anchors_nb
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5 # 5 = bx,by,bh,bw,pc
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2]) # x, y
    netout[..., :2] = netout[..., :2]*scales_x_y - 0.5*(scales_x_y - 1.0) # scale x, y

    netout[..., 4:] = _sigmoid(netout[..., 4:]) # objectness + classes probabilities

    for i in range(grid_h*grid_w):

        row = i / grid_w
        col = i % grid_w
        
        
        for b in range(nb_box):
            # 4th element is objectness
            objectness = netout[int(row)][int(col)][b][4]

            if(objectness > obj_thresh):
                #print("objectness: ",objectness)                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height            
                # last elements are class probabilities
                classes = objectness*netout[int(row)][col][b][5:]
                classes *= classes > obj_thresh
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)           
                boxes.append(box)
    return boxes

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] ), int(x[1] ), int(x[2] )), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# get all of the results above a threshold (Edited to suit detection of Cell Phones only)
def get_boxes(boxes, labels, thresh, colors):
    v_boxes, v_labels, v_scores, v_colors = list(), list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        if box.classes[cellphone_idx] > thresh:
            v_boxes.append(box)
            v_labels.append(labels[cellphone_idx])
            v_scores.append(box.classes[cellphone_idx]*100)
            v_colors.append(colors[cellphone_idx])
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores, v_colors
class_threshold = 0.65

# Main Yolo Inference for loop
def Inference(image,input_w,input_h,colors,labels,phoneFramesTotal):

   
    # Get Dimentions of resized frame image.
    # Run the model

    yhat = yolo_model.predict(image)

    # Compute the Yolo layers
    obj_thresh = 0.55
    anchors = [ [12, 16, 19, 36, 40, 28],[36, 75, 76, 55, 72, 146],[142, 110, 192, 243, 459, 401]]
    scales_x_y = [1.2, 1.1, 1.05]
    boxes = list()

    for i in range(len(anchors)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], obj_thresh, input_h, input_w, len(anchors), scales_x_y[i])

    
    # Correct the boxes according the inital size of the image and Do NMS
    #correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    do_nms(boxes, 0.38)

    
    # Final Boxes
    v_boxes, v_labels, v_scores, v_colors = get_boxes(boxes, labels, 0.5, colors)
    
    ## Return image as is if no boxes found #####
    if len(v_labels) == 0:
        return np.reshape(image,image.shape[1:]),phoneFramesTotal
    
    ## Possible Redudant if statement since we directly only pick Cellphone idx . Needs to be checked
    if 'cell phone' in v_labels:
        phoneFramesTotal +=1
    
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = int((box.ymin)*608), int((box.xmin)*608), int((box.ymax)*608), int((box.xmax)*608)
        
        if len(image.shape) == 4:
            image = np.squeeze(image,axis = 0)

        image = cv2.rectangle(image, (x1, y2),(x2, y1),v_colors[i], 2)
       
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        image = cv2.putText(image, label, (x1,y1 -5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(v_colors[i]), 2)
        
    return image,phoneFramesTotal

########## Main Face Recognition inference for loop ##########
def faceRecInference(faceEncodingsKnown,faceNames,frame,absentFramesTotal,nameToCheck):
    
    
    # Resize Frame to half for better performance and convert from BGR 2 RGB
    frameS = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frameS = frameS[:, :, ::-1]
    faceLocsCurr = faceRec.face_locations(frameS)
    faceEncsCurr = faceRec.face_encodings(frameS, faceLocsCurr)
    faceNamesInFrame = []
        
    for faceEnc in faceEncsCurr:
        ### Initialize with Unknown face and replace if match found ######
        name = 'Unknown Face'
        matches = faceRec.compare_faces(faceEncodingsKnown,faceEnc,tolerance = 0.41) # 0.41 strictness for face recogntion since i found this as a good balance. Needs more testing however
        dist = faceRec.face_distance(faceEncodingsKnown,faceEnc)
        bestMatchIdx = np.argmax(dist)
        
        if matches[bestMatchIdx]:
            name = faceNames[bestMatchIdx]
        faceNamesInFrame.append(name)

    #### Check if Student to Check for is the only one giving the exam and not some one else #####
    if nameToCheck not in faceNamesInFrame:
        absentFramesTotal +=1
        CenterPos = ((int) (frame.shape[1]/2 - 268/2 + 15), (int) (frame.shape[0]/2 - 36/2) + 160)
        cv2.putText(frame, 'Student Missing',CenterPos, cv2.FONT_HERSHEY_TRIPLEX, 1.25, (0,0,255), 1)
        

    for (top,right,bottom,left),name in zip(faceLocsCurr,faceNamesInFrame):
        # Scaling Bounding boxes back to fit orignal image
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

         # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    return frame,absentFramesTotal

class WrongBoolVal(Exception):
    pass

def getFrameSec(fps):
    fSec = 100/(fps*100)
    return fSec


