import numpy as np
import cv2

def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def unpack_labeling_spec(spec): 
    keys = list(spec.keys())

    for key in spec: 
        keys.extend(unpack_labeling_spec(spec[key]))
    return keys


def draw_labels_on_image(image, startx, labeling_spec, label_probs): 
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1.0
    lineType               = 2
    starty = 30

    keys = labeling_spec.keys()
    values = [label_probs[key] for key in keys]
    values = softmax(values)
    
    maxindex = np.argmax(values)
    
    for index, key in enumerate(keys): 
        bottomLeftCornerOfText = (startx, starty)

        if index == maxindex: 
            fontColor = (255,255,0)
        else: 
            fontColor = (255,0,255)

        image = cv2.putText(image,
                            key + " " + "{:.2f}".format(values[index]), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        
        starty += 30
        
        if index == maxindex and len(labeling_spec[key].keys()) > 0:
            
            image = draw_labels_on_image(image, startx + 300, labeling_spec[key], label_probs)
        
    return image