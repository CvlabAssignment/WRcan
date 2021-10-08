from PIL import Image
import xml.etree.ElementTree as ET
import re
import glob

file_path = '/data1/KIST/data/20200904_KIST_DATA/'
# file_names = glob.glob(file_path+'*_FrameInfo.xml')
# file_names2 = []

images_path = './images_ex/'
# images_file_names = glob.glob(images_path+'*.jpeg')
# images_file_names = glob.glob(images_path+'카메라01_*.jpeg')
images_file_names = ['카메라01_20200904170300_20200904172800_0|24602.jpeg']
images_frame = []

for f in images_file_names:
    # f = f.split("/")
    f = f.split("|")
    frame_no = int(f[1][:-5])
    # frame_no = 24602
    file_name = f[0]
    # print(frame_no, file_name)
    tree = ET.parse(file_path + '{}_FrameInfo.xml'.format(file_name))
    root = tree.getroot()
    child = root[0]
    print(child[frame_no-1].attrib, frame_no, file_name)
    for i in child[frame_no-1]:
        if i.tag.startswith('H'):
            # print('H',i.tag)
            numbers = re.findall(r'\d+', str(i.attrib))
            # print('n',numbers)
            frame_no = 24602
            img = Image.open(images_path + file_name + "|" + str(frame_no) + ".jpeg")
            # print(i.attrib, numbers)
            l = int(numbers[1])
            t = int(numbers[3])
            r = int(numbers[5])
            b = int(numbers[7])
            # print(img_idx, area)
            cropped_img = img.crop((l, t, r, b))
            cropped_img.save('./images_crop_test/' +str(file_name) + "_" + str(frame_no) + "_" + str(i.tag) + '.jpg')

# images_x1_path = './images_x1/'
# images_x1_file_names = glob.glob(images_x1_path+'*.jpeg')
# images_x1_frame = []

# for f in images_x1_file_names:
#     f = f.split("/")
#     f = f[2].split("|")
#     frame_no = int(f[1][:-5])
#     file_name = f[0][:-3]
#     print(frame_no, file_name)
#     tree = ET.parse(file_path + '{}_FrameInfo.xml'.format(file_name))
#     root = tree.getroot()
#     child = root[0]
#     print(child[frame_no - 1].attrib, frame_no, file_name)
#     for i in child[frame_no -1]:
#         if i.tag.startswith('H'):
#             numbers = re.findall(r'\d+', str(i.attrib))
#             # print('n',numbers)
#             img = Image.open("./images/" + file_name + "|" + str(frame_no) + ".jpeg")
#             # print(i.attrib, numbers)
#             l = int(numbers[1])
#             t = int(numbers[3])
#             r = int(numbers[5])
#             b = int(numbers[7])
#             # print(img_idx, area)
#             cropped_img = img.crop((l, t, r, b))
#             cropped_img.save('./images_x1_crop/' +str(file_name) + "_" + str(frame_no) + "_" + str(i.tag) + '.jpg')


