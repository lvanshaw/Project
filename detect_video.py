import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import coreD.utils as utils
from coreD.yolov4 import filter_boxes
from coreD.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np

def process_video(framework='tf', weights='./checkpoints/yolov4-416', size=416, tiny=False, model='yolov4', video='', output=None, output_format='XVID', iou=0.45, score=0.50, count=False, dont_show=False, info=True, crop=True, plate=True):

    input_size = size
    video_path = video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        print(saved_model_loaded)
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame_num = 0
    # initialize a counter for the box IDs
    box_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
         # create a list to hold the box IDs
        box_ids = []

        # iterate through the detected boxes and append an ID to the list
        for i in range(valid_detections[0]):
            box_ids.append(box_id)
            box_id += 1
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        for i in range(valid_detections[0]):
            box = boxes[0][i]
            score = scores[0][i]
            label = classes[0][i]
            box_id = box_ids[i]
            print('Box ID:', box_id, 'Label:', label.numpy(), 'Score:', score.numpy(), 'Box:', box.numpy())
            
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        # if crop:
        #     crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
        #     crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
        #     try:
        #         os.mkdir(crop_path)
        #     except FileExistsError:
        #         pass
        #     if frame_num % crop_rate == 0:
        #         final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
        #         try:
        #             os.mkdir(final_path)
        #         except FileExistsError:
        #             pass          
        #         crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
        #     else:
        #         pass
        if crop:
            for i in range(valid_detections[0]):
                box = boxes[0][i]
                score = scores[0][i]
                label = classes[0][i]
                box_id = box_ids[i]
                print('Box ID:', box_id, 'Label:', label.numpy(), 'Score:', score.numpy(), 'Box:', box.numpy())
                if allowed_classes[int(label)] in allowed_classes:
                    # Get the coordinates of the bounding box
                    ymin, xmin, ymax, xmax = box.numpy()
                    # Get the width and height of the bounding box
                    width = xmax - xmin
                    height = ymax - ymin
                    # Scale the bounding box coordinates to the original image size
                    xmin = int(xmin * original_w)
                    xmax = int(xmax * original_w)
                    ymin = int(ymin * original_h)
                    ymax = int(ymax * original_h)
                    # Crop the image using the bounding box coordinates
                    cropped_image = frame[ymin:ymax, xmin:xmax]
                    # Save the cropped image with the box ID as the filename
                    cv2.imwrite(f'./data/output/{video_name}/box{box_id}.jpg', cropped_image)
        if count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, info, counted_classes, allowed_classes=allowed_classes, read_plate=plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, info, allowed_classes=allowed_classes, read_plate=plate)
        
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not dont_show:
            cv2.imshow("result", result)
        
        if output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


# python detect_video.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --crop
