
import cv2
import pandas as pd
from PIL import Image
IMAGE_BASE_DIR = 'D://DASH2'

full_labels = pd.read_csv('data/light_labels.csv')

print(full_labels.head())


def draw_boxes(image_name):
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread(IMAGE_BASE_DIR + '//{}'.format(image_name))
    for index, row in selected_value.iterrows():
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 3)
    return img


Image.fromarray(draw_boxes('dash_164_.jpg'))


Image.fromarray(draw_boxes('xflsaiou15.jpg'))