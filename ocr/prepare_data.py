import glob
import logging
import os
import xml.etree.ElementTree as ET

import cv2
import pandas as pd

# Create and configure logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def parse_xml(file):
    root_element = ET.parse(file).getroot()  # annotation tag

    # image info
    filename = root_element.find("filename").text  # filename value
    # getting base directory of images
    base_dir = "/".join(file.split('/')[:-1])
    # appending directory with filename to get abs path
    filename = os.path.join(base_dir, filename)

    width = root_element.find("size/width").text  # width value
    height = root_element.find("size/height").text  # height value
    depth = root_element.find("size/depth").text  # depth value

    # bounding box info
    xmin = root_element.find("object/bndbox/xmin").text  # xmin value
    ymin = root_element.find("object/bndbox/ymin").text  # ymin value
    xmax = root_element.find("object/bndbox/xmax").text  # xmax value
    ymax = root_element.find("object/bndbox/ymax").text  # ymax value

    number = root_element.find("object/name").text  # number

    values_extracted = {
        "filename": [filename],
        "width": [width],
        "height": [height],
        "depth": [depth],
        "xmin": [xmin],
        "ymin": [ymin],
        "xmax": [xmax],
        "ymax": [ymax],
        "number": [number]
    }

    return values_extracted


def read_all_xmls(root_dir):
    r"""
    returns dataframe with info about image filenames and bounding boxes

    Args:
        root_dir: root directory where data in locates
    """

    # list all xml files
    xml_files = glob.glob(os.path.join(root_dir, "data/*/*.xml"))

    # empty dataframe with only column names
    columns = [
        "filename",
        "width",
        "height",
        "depth",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "number"
    ]

    df = pd.DataFrame(columns=columns)

    # extract file name and bounding boxes
    logging.info('reading xmls files...')
    for file in xml_files:
        new_data = parse_xml(file)
        temp = pd.DataFrame.from_dict(new_data)
        df = pd.concat([df, temp])
    logging.info('completed reading xml files')

    # reseting the index field in dataframe
    df.reset_index(drop=True, inplace=True)
    return df


def create_number_plate_dataset(root_dir):
    dataset_path = os.path.join(root_dir, 'ocr_data')
    os.makedirs(dataset_path, exist_ok=True)

    # read and crop number plate portion and save the file with name of
    # the file as the label of the image

    df = read_all_xmls(root_dir)

    logging.info('creating ocr dataset...')
    for entry in df.iterrows():
        *bbox, number = entry[1][-5:]
        xmin, ymin, xmax, ymax = list(map(int, bbox))
        image = cv2.imread(entry[1][0])[ymin: ymax, xmin: xmax]
        cv2.imwrite(os.path.join(dataset_path, f"{number}.jpg"), image)
    logging.info('completed dataset creations')
