import pandas as pd
import scipy
import numpy as np
  
def read_txt(path):
    with open(path, 'r') as f:
        tmp = f.read().splitlines()
        head = tmp[0]
        contx = tmp[1:]
    return contx, head
 
def write_txt(path, contx):
    np.savetxt(path, contx, fmt='%s', delimiter=',')

if __name__ == "__main__":
    path = "./pa100k"
    labels, labels_head = read_txt(path+"/labels.txt")
    images, images_head = read_txt(path+"/images.txt")
    np.random.seed(42)
    np.random.shuffle(images) 
    np.random.seed(42)
    np.random.shuffle(labels)
    train_labels = labels[:80000]
    train_images = images[:80000]
    val_labels = labels[80000:90000]
    val_images = images[80000:90000]
    test_labels = labels[90000:]
    test_images = images[90000:]
    train_labels.insert(0, labels_head)
    train_images.insert(0, images_head)
    val_labels.insert(0, labels_head)
    val_images.insert(0, images_head)
    test_labels.insert(0, labels_head)
    test_images.insert(0, images_head)
    print(len(train_labels),len(train_images),len(val_labels),len(val_images),len(test_labels),len(test_images))
    write_txt(path+"/train_label.txt", train_labels)
    write_txt(path+"/train_images_name.txt", train_images)
    write_txt(path+"/val_label.txt", val_labels)
    write_txt(path+"/val_images_name.txt", val_images)
    write_txt(path+"/test_label.txt", test_labels)
    write_txt(path+"/test_images_name.txt", test_images)

