#Reference: http://blog.csdn.net/u014365862/article/details/53869837
#Reference: Deep Convolutional Network for Handwritten Chinese Character Recognition

import os
import numpy as np
import struct
import PIL.Image

train_data_dir = "HWDB1.1trn_gnt"
test_data_dir = "HWDB1.1tst_gnt"

def read_from_gnt_dir(gnt_dir = train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f,dtype='uint8', count=width*height).reshape((height,width))
            yield image, tagcode
            
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode
                    
import scipy.misc
from sklearn.utils import shuffle
import tensorflow as tf

char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

def resize_and_normalize_image(img):
    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0,0))
    else:
        pad_dims = ((0,0), (pad_size,pad_size))
    img = np.lib.pad(img, pad_dims, mode = 'constant', constant_values = 255)
    
    img = scipy.misc.imresize(img, (64-4*2, 64-4*2))
    img = np.lib.pad(img, ((4,4),(4,4)), mode = 'constant', constant_values=255)
    assert img.shape == (64,64)
    
    img = img.flatten()
    img = (img - 128)/128
    return img
    
def convert_to_one_hot(char):
    vector = np.zeros(len(char_set))
    vector[char_set.index(char)] = 1
    return vector
    
train_data_x = []
train_data_y = []

for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    if tagcode_unicode in char_set:
        train_data_x.append(resize_and_normalize_image(image))
        train_data_y.append(convert_to_one_hot(tagcode_unicode))
        
train_data_x, train_data_y = shuffle(train_data_x, train_data_y, random_state=0)

batch_size = 128
num_batch = len(train_data_x) // batch_size

text_data_x = []
text_data_y = []
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    if tagcode_unicode in char_set:
        text_data_x.append(resize_and_normalize_image(image))
        text_data_y.append(convert_to_one_hot(tagcode_unicode))

text_data_x, text_data_y = shuffle(text_data_x, text_data_y, random_state=0)

X = tf.placeholder(tf.float32, [None, 64*64])
Y = tf.placeholder(tf.float32, [None, 140])
keep_prob = tf.placeholder(tf.float32)

def chinese_hand_write_cnn():
    x = tf.reshape(X, shape = [-1,64,64,1])
    
    w_c1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
    b_c1 = tf.Variable(tf.zeros([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1,1,1,1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    w_c2 = 