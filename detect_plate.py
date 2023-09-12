from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import imutils
import cv2
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from function import *

def predic(filen,file):
    filename = './data/'+file

    filetype = file.split('.')[-1]
    if filetype in ['jpg', 'jpeg', 'png']:
        car_image = imread(filename, as_gray=True)
    else:
        if os.path.exists('output'):
            shutil.rmtree('output')

        os.makedirs('output')

        cap = cv2.VideoCapture(filename)
        # cap = cv2.VideoCapture(0)
        count = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret == True:
                cv2.imshow('window-name',frame)
                cv2.imwrite("./output/frame%d.jpg" % count, frame)
                count = count + 1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        car_image = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
        car_image = imutils.rotate(car_image, 270)
    print(car_image.shape)


    gray_car_image = car_image * 255
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(gray_car_image, cmap="gray")
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    print(binary_car_image)
    ax2.imshow(binary_car_image, cmap="gray")
    # ax2.imshow(gray_car_image, cmap="gray")
    plt.show()
    label_image = measure.label(binary_car_image)

    plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
    plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")
    flag =0

    for region in regionprops(label_image):
        # print(region)
        if region.area < 50:
            continue

        min_row, min_col, max_row, max_col = region.bbox

        region_height = max_row - min_row
        region_width = max_col - min_col


        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            flag = 1
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                    min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                            max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                        linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions
    if(flag == 1):
        # print(plate_like_objects[0])
        plt.show()




    if(flag==0):
        min_height, max_height, min_width, max_width = plate_dimensions2
        plate_objects_cordinates = []
        plate_like_objects = []

        fig, (ax1) = plt.subplots(1)
        ax1.imshow(gray_car_image, cmap="gray")


        for region in regionprops(label_image):
            if region.area < 50:
                continue

            min_row, min_col, max_row, max_col = region.bbox


            region_height = max_row - min_row
            region_width = max_col - min_col

            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                # print("hello")
                plate_like_objects.append(binary_car_image[min_row:max_row,
                                        min_col:max_col])
                plate_objects_cordinates.append((min_row, min_col,
                                                max_row, max_col))
                rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                            linewidth=2, fill=False)
                ax1.add_patch(rectBorder)
                # let's draw a red rectangle over those regions
        # print(plate_like_objects[0])
        plt.show()







    # The invert was done so as to convert the black pixel to white pixel and vice versa
    license_plate = np.invert(plate_like_objects[0])

    labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")

    character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter=0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]

            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                        linewidth=2, fill=False)
            ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)
    # print(characters)
    plt.show()


    import pickle
    print("Loading model")
    filename = './finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    print('Model loaded. Predicting characters of number plate')
    classification_result = []
    for each_character in characters:
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        result = model.predict(each_character)
        classification_result.append(result)

    print('Classification result')
    print(classification_result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    print('Predicted license plate')
    print(plate_string)

    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]
    print("DETECTION AUTO: "+filen)
    #detection_auto(filen)
    print('License plate')
    print(rightplate_string)
    return rightplate_string
    #update_json(filen, 'detection',rightplate_string)
    
def update_json(filename, result_type, result):
    uploads = load_uploads()

    for user in uploads:
        for upload in uploads[user]:
            if upload['filename'] == filename:
                if(result_type == 'speed'):
                    upload['speed_clicked'] = True
                    upload[result_type] = result
                if(result_type == 'detection'):
                    upload['detection_clicked'] = True
                    upload[result_type] = result

    with open(JSON_FILE, 'w') as f:
        json.dump(uploads, f, indent=2)


