'''
This code from google deep learning course
'''

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import random
import time 
import hashlib
import multiprocessing.pool as pool
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

# six模块兼容2.x与3.x
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle # 序列化与反序列化

image_size = 28 # Pixel width and height
pixel_depth = 255.0 # Number of levels per pixel.

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '/tmp/data/notmnist' # Change me to store data elsewhere

num_classes = 10
np.random.seed(133)

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
      if percent % 5 == 0:
          sys.stdout.write("%s%%" % percent)
          sys.stdout.flush()
      else:
          sys.stdout.write(".")
          sys.stdout.flush()
      
      last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
          'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename

def maybe_extract(filename, force=False):
    """解压文件"""
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
      os.path.join(root, d) for d in sorted(os.listdir(root))
      if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
          'Expected %d folders, one per class. Found %d instead.' % (
            num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                            dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            # 将每个图像转化为二维数组，将二维数组的值去均值并归一化到[-1, +1]范围内
            # ndimage.imread将一个图片文件读取为二维数组，并且数组长度为图片像素
            # image_data.shape为数组维度（此处即为像素）
            # 参考numpy运算
            image_data = (ndimage.imread(image_file).astype(float) - 
                            pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                # throw an exception
                raise Exception("Unexcepted image shape: %s" % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print("Could not read:", image_file, ":", e, "- it\'s ok, skipping.")
    
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception("Many fewer images than expected: %d < %d" % 
                        (num_images, min_num_images))

    print("Full dataset tensor:", dataset.shape)
    print("Mean:", np.mean(dataset))
    print("Standard deviation:", np.std(dataset))

    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    """将文件信息集合序列化储存在.pickle文件中，并返回储存后的文件名列表"""
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + ".pickle"
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print("%s already present - Skipping pickling." % set_filename)
        else:
            print("Picking %s" % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, "wb") as f:
                    # 序列化
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("Unable to save data to", set_filename, ":", e)

    return dataset_names

def show_pickle(train_datasets):
    """随机绘制图像"""
    fig = plt.figure()
    for i in range(10):
        pickle_file = train_datasets[i]
        with open(pickle_file, "rb") as f:
            # 反序列化dataset（读取保存的数据文件）
            letter_set = pickle.load(f)
            # 在dataset的数量范围内产生随机数，并绘制以这个随机数为索引的图像
            sample_idx = np.random.randint(len(letter_set))
            sample_image = letter_set[sample_idx, :, :]
            # 显示多个图像1行10列
            fig.add_subplot(1, 10, i + 1)
            plt.imshow(sample_image)
    plt.show()

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    """分离出训练集与验证集"""
    # 分类数量
    num_classes = len(pickle_files)
    # 验证集
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    # 训练集
    train_dataset, train_labels = make_arrays(train_size, image_size)

    # //表示整数除法，/表示浮点数除法
    # 使用//时需要导入模块__future__
    # 每一个分类的训练集与验证集大小
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class

    # python内置枚举函数，同时获取索引和值
    # 遍历数据集
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, "rb") as f:
                # letter 字母
                letter_set = pickle.load(f)

                # lte's shuffle the letters to have random validation and training set
                # 用于将一个序列中数据打乱
                np.random.shuffle(letter_set)

                # 打乱后每组取前vsize_per_class进行合并，合并为训练集与验证集
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label # 索引
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class

        except Exception as e:
            print("Unable to process data from", pickle_file, ":", e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
    """打乱数据集数据"""
    # 产生一个乱序数组
    permutation = np.random.permutation(labels.shape[0])
    shuffed_dataset = dataset[permutation, :, :]
    shuffed_labels = labels[permutation]
    return shuffed_dataset, shuffed_labels

def plot_sample_dataset(dataset, labels, title):
    """显示打乱后的图像"""
    plt.figure()
    plt.suptitle(title, fontsize=16, fontweight="bold")
    items = random.sample(range(len(labels)), 12)
    for i, item in enumerate(items):
        plt.subplot(3, 4, i + 1)
        plt.axis("off")
        plt.title(chr(ord("A") + labels[item]))
        plt.imshow(dataset[item])
    plt.show()

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

# show_pickle(train_datasets)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(
    test_datasets, test_size)

# 打印训练集/验证集/测试集的信息
print("Training:", train_dataset.shape, train_labels.shape)
print("Validation:", valid_dataset.shape, valid_labels.shape)
print("Testing:", test_dataset.shape, test_labels.shape)

# 打乱数据
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# 储存足够乱序的训练集/验证集/测试集
pickle_file = os.path.join(data_root, "notMNIST.pickle")

try:
    f = open(pickle_file, "wb")
    save = {
        "train_dataset": train_dataset,
        "train_labels": train_labels,
        "valid_dataset": valid_dataset,
        "valid_labels": valid_labels,
        "test_dataset": test_dataset,
        "test_labels": test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close
except Exception as e:
    print("Unable to save data to", pickle_file, ":", e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

plot_sample_dataset(train_dataset, train_labels, 'train dataset suffled')
plot_sample_dataset(valid_dataset, valid_labels, 'valid dataset suffled')
plot_sample_dataset(test_dataset, test_labels, 'test dataset suffled')


def display_overlap(overlap, source_dataset, target_dataset):
    """显示重复部分图案"""
    # 在python3.x中，dict.keys()返回一个dict_keys对象，比起列表，这个对象的行为更像是set
    # 所以不支持索引的。解决方案：list(dict.keys())
    item = random.choice(list(overlap.keys()))
    imgs = np.concatenate(([source_dataset[item]], target_dataset[overlap[item][0:7]]))
    plt.figure()
    plt.suptitle(item)
    for i, img in enumerate(imgs):
        plt.subplot(2, 4, i+1)
        plt.axis("off")
        plt.imshow(img)
    plt.show()

def extract_overlap(dataset_1, dataset_2):
    """查找重复图案（通过对比数组是否相同）"""
    overlap = {}
    for i, img_1 in enumerate(dataset_1):
        for j, img_2 in enumerate(dataset_2):
            if np.array_equal(img_1, img_2):
                if not i in overlap.keys():
                    overlap[i] = []
                overlap[i].append(j)
    
    return overlap

def extract_overlap_hash(dataset_1, dataset_2):
    """查找重复图案（通过hash查找）"""
    overlap = {}
    dataset_hash_1 = dataset_1
    dataset_hash_2 = dataset_2
    for i, hash1 in enumerate(dataset_hash_1):
        for j, hash2 in enumerate(dataset_hash_2):
            if hash1 == hash2:
                if not i in overlap.keys():
                    overlap[i] = []
                overlap[i].append(j)
    
    return overlap

def sanetize(dataset_1, dateset_2, labels_1):
    """去除重复"""
    overlap = []
    for i, hash1 in enumerate(dataset_1):
        # np.where 类似三目运算符，有三个参数，第一个为条件表达式
        # 第二与第三个参数可选，当在数组某一位置条件为True时，
        # 输出x的对应位置的元素，否则选择y对应位置的元素；
        # 如果后两个参数为None，则返回为True的坐标位置信息
        duplicates = np.where(hash1 == dateset_2)
        if len(duplicates[0]):
            overlap.append(i)
    return np.delete(dataset_1, overlap, 0), np.delete(labels_1, overlap, None)

# 计算数据集的hash值
def cacl_hash(dataset):
    print("start cacl hash")
    return [hashlib.sha256(img).hexdigest() for img in dataset]

test_dataset_hash_1 = cacl_hash(test_dataset)
train_dataset_hash_2 = cacl_hash(train_dataset)

print("test dataset shape:", test_dataset.shape)
print('train dataset shape:', train_dataset.shape)
print("valid dataset shape:", valid_dataset.shape)

print("cacl hash")
test_dataset_hash_1 = cacl_hash(test_dataset)
train_dataset_hash_2 = cacl_hash(train_dataset)
valid_dataset_hash = cacl_hash(valid_dataset)

start_time = time.time()
overlap_test_train = extract_overlap_hash(test_dataset_hash_1, train_dataset_hash_2)
end_time = time.time()
print("start time: ", start_time)
print("end time: ", end_time)
print("use time: ", start_time - end_time)

print('Number of overlaps:', len(overlap_test_train.keys()))
display_overlap(overlap_test_train, test_dataset, train_dataset)

start_time = time.time()
test_dataset_sanit, test_labels_sanit = sanetize(test_dataset_hash_1, train_dataset_hash_2, test_labels)
print("Overlapping images removed: " , len(test_dataset) - len(test_dataset_sanit))
end_time = time.time()
print("start time: ", start_time)
print("end time: ", end_time)
print("use time: ", start_time - end_time)

valid_dataset_sanit, valid_labels_sanit = sanetize(valid_dataset_hash, train_dataset_hash_2, valid_labels)

# 储存去重后的文件
pickle_file_sanit = os.path.join(data_root, "notMNIST_sanit.pickle")

try:
    f = open(pickle_file_sanit, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset_sanit,
        'valid_labels': valid_labels_sanit,
        'test_dataset': test_dataset_sanit,
        'test_labels': test_labels_sanit,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file_sanit)
print('Compressed pickle size:', statinfo.st_size)



print("save ndarray")
train_dataset_file_sanit = os.path.join(data_root, "notMNIST_train_dataset_sanit.pickle")
train_labels_file_sanit = os.path.join(data_root, "notMNIST_train_labels_sanit.pickle")
valid_dataset_file_sanit = os.path.join(data_root, "notMNIST_valid_dataset_sanit.pickle")
valid_labels_file_sanit = os.path.join(data_root, "notMNIST_valid_labels_sanit.pickle")
test_dataset_file_sanit = os.path.join(data_root, "notMNIST_test_dataset_sanit.pickle")
test_labels_file_sanit = os.path.join(data_root, "notMNIST_test_labels_sanit.pickle")

np.save(train_dataset_file_sanit, train_dataset)
np.save(train_labels_file_sanit, train_labels)
np.save(valid_dataset_file_sanit, valid_dataset_sanit)
np.save(valid_labels_file_sanit, valid_labels_sanit)
np.save(test_dataset_file_sanit, test_dataset_sanit)
np.save(test_labels_file_sanit, test_labels_sanit)

print("end save")


def disp_sample_dataset(dataset, labels, title=None):
    fig = plt.figure()
    if title: fig.suptitle(title, fontsize=16, fontweight='bold')
    items = random.sample(range(len(labels)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[item]))
        plt.imshow(dataset[item])
    plt.show()


def train_and_predict(sample_size):
    regr = LogisticRegression()
    X_train = train_dataset[:sample_size].reshape(sample_size, 784)
    y_train = train_labels[:sample_size]
    regr.fit(X_train, y_train)

    X_test = test_dataset.reshape(test_dataset.shape[0], 28 * 28)
    y_test = test_labels

    pred_labels = regr.predict(X_test)

    print('Accuracy:', regr.score(X_test, y_test), 'when sample_size=', sample_size)
    disp_sample_dataset(test_dataset, pred_labels, 'sample_size=' + str(sample_size))


for sample_size in [50, 100, 1000, 5000, len(train_dataset)]:
    train_and_predict(sample_size)
