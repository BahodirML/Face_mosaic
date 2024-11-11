from facedetect.facedetect import *
from mosaic import *
import os
import time
import cv2
from cv2 import dnn
import numpy as np

# Hardcoded paths for convenience
ONNX_PATH = "C:/Users/bahod/Desktop/MosaicFace/Automatic_Face_Mosaic/facedetect/version-RFB-320_simplified.onnx"
INPUT_PATH = "C:/Users/bahod/Desktop/MosaicFace/Automatic_Face_Mosaic/imgs"  # Set your folder path here
RESULTS_PATH = "C:/Users/bahod/Desktop/MosaicFace/Automatic_Face_Mosaic/results"
INPUT_SIZE = (320, 240)
THRESHOLD = 0.2
GRAD_SIZE = 8

def preprocess_image(image, target_size=(640, 480)):
    h, w = image.shape[:2]
    scaling_factor = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, scaling_factor

def process_image(img_path, net, priors, scaling_factor):
    img_ori = cv2.imread(img_path)
    preprocessed_img, scale = preprocess_image(img_ori)
    rect = cv2.resize(preprocessed_img, INPUT_SIZE)
    rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
    
    net.setInput(dnn.blobFromImage(rect, 1 / image_std, INPUT_SIZE, 127))
    boxes, scores = net.forward(["boxes", "scores"])
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = center_form_to_corner_form(boxes)
    
    boxes, labels, probs = predict(preprocessed_img.shape[1], preprocessed_img.shape[0], scores, boxes, THRESHOLD)
    boxes = (boxes / scale).astype(int)

    for box in boxes:
        mosaic(img_ori, (box[0], box[1]), (box[2] - box[0], box[3] - box[1]), GRAD_SIZE)
    output_path = os.path.join(RESULTS_PATH, 'mosaic_' + os.path.basename(img_path))
    cv2.imwrite(output_path, img_ori)
    print(f"Processed image saved to {output_path}")

def process_video(video_path, net, priors):
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join(RESULTS_PATH, 'mosaic_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame, scale = preprocess_image(frame)
        rect = cv2.resize(preprocessed_frame, INPUT_SIZE)
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        
        net.setInput(dnn.blobFromImage(rect, 1 / image_std, INPUT_SIZE, 127))
        boxes, scores = net.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = center_form_to_corner_form(boxes)
        
        boxes, labels, probs = predict(preprocessed_frame.shape[1], preprocessed_frame.shape[0], scores, boxes, THRESHOLD)
        boxes = (boxes / scale).astype(int)

        for box in boxes:
            mosaic(frame, (box[0], box[1]), (box[2] - box[0], box[3] - box[1]), GRAD_SIZE)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (frame.shape[1], frame.shape[0]))
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def mosaicFaces():
    net = dnn.readNetFromONNX(ONNX_PATH)
    priors = define_img_size(INPUT_SIZE)

    if os.path.isdir(INPUT_PATH):
        files = os.listdir(INPUT_PATH)
        for file in files:
            file_path = os.path.join(INPUT_PATH, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    process_image(file_path, net, priors, scaling_factor=1.0)
                elif file.lower().endswith(('.mp4', '.avi', '.mov')):
                    process_video(file_path, net, priors)
    else:
        if INPUT_PATH.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(INPUT_PATH, net, priors, scaling_factor=1.0)
        elif INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(INPUT_PATH, net, priors)

if __name__ == '__main__':
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    mosaicFaces()
