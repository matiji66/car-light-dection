import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# 'annotations'
ANNOTATIONS_PATH = 'D://DASH2'


def xml_to_csv(path, shuffle=True):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    if shuffle :
        import numpy as np
        xml_df = xml_df.iloc[np.random.permutation(len(xml_df))]
    return xml_df


def main():

    image_path = os.path.join(os.getcwd(), ANNOTATIONS_PATH)
    xml_df = xml_to_csv(image_path, shuffle=False)
    xml_df.to_csv('data/light_labels.csv', index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()