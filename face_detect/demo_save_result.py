# MoboleNetssd 和 于仕琪人脸检测网络推理结果，保存结果为xmin,ymin,xmax,ymax,conf  或着 xmin,ymin,w,h,conf

import numpy as np  
import sys,os  
import cv2
#caffe_root = '/media/ryan/F/Projects/caffe/'
#sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file= '/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/mobilenet_ssd_t1_hub-master/graphs/1227/face_deploy_8.prototxt'
caffe_model='/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/mobilenet_ssd_t1_hub-master/graphs/1227/60000_8_focal_g1_bgs_test.caffemodel'
test_dir = "/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/rgb_image"
# net_file = "/home/ljdong/code/python_code/libfacedetection/models/caffe/yufacedetectnet-open-v1.prototxt"
# caffe_model = "/home/ljdong/code/python_code/libfacedetection/models/caffe/yufacedetectnet-open-v1.caffemodel"
if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background','face')


def preprocess(src):
    img = cv2.resize(src, (320,240))
    image = src.copy()
    print(img.shape)
    img = img- 0.0
    img[:,:,0] = img[:,:,0] - 104
    img[:,:,1] = img[:,:,1] - 117
    img[:,:,2] = img[:,:,2] - 123
    # img = img *  0.0039062
    return img, image

def preprocess_mssd(src):
    image = src.copy()
    img = cv2.resize(src, (256,256))
    img = img -127.5
    img = img * (1.0/127.5)
    return img, image

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1] 
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def postprocess_original(out,w,h):   
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile, fddb = False):
    detect_result = []
    if fddb:
        detect_result.append('/'.join(imgfile.split('/')[-5:]))
    else:
        detect_result.append('/'.join(imgfile.split('/')[-2:]))
    detect_result.append("\n")
    if fddb:
        imgfile += '.jpg'
    origimg = cv2.imread(imgfile)
    print(imgfile)
    h, w = origimg.shape[:2]
    # img, origimg = preprocess(origimg)
    img, origimg = preprocess_mssd(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    # box, conf, cls = postprocess(origimg, out)
    box, conf, cls = postprocess_original(out,w,h)
    # print("imagefile: ",imgfile)
    detect_result.append(len(box))
    detect_result.append("\n")
    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       if not fddb:
           detect_result.append(box[i][0])
           detect_result.append(' ')
           detect_result.append(box[i][1])
           detect_result.append(' ')
           detect_result.append(box[i][2])
           detect_result.append(' ')
           detect_result.append(box[i][3])
           detect_result.append(' ')
           detect_result.append(conf[i])
           detect_result.append('\n')
       else:
           detect_result.append(box[i][0])
           detect_result.append(' ')
           detect_result.append(box[i][1])
           detect_result.append(' ')
           detect_result.append(box[i][2] - box[i][0])
           detect_result.append(' ')
           detect_result.append(box[i][3] - box[i][1])
           detect_result.append(' ')
           detect_result.append(conf[i])
           detect_result.append('\n')
       print("(",box[i][0],",", box[i][1],")",",","(",box[i][2],",", box[i][3],")")
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("ssd", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return detect_result

def get_img_file(filetxt,root_path):
    file_list = []
    with open(filetxt,'r') as r:
        imgfile = r.read().split('\n')
    for name in imgfile:
        file_list.append(root_path + '/' + name)
    return file_list    

def main():
    result_save = "/home/ljdong/data/wilderface/" + net_file.split('/')[-1] + 'fddb.txt'
    with open(result_save,'w') as w:
        for imgfile in get_img_file("/home/ljdong/PycharmProjects/facedetect/MobilenetSSDFace/FDDBfacedet/FDDB-folds/fold-all.txt","/home/ljdong/PycharmProjects/facedetect/MobilenetSSDFace/FDDBfacedet"):
            img_result = detect(imgfile,True)
            for info in img_result:
                w.write(str(info))
def main_show():
    for imgfile in get_img_file("/home/ljdong/PycharmProjects/facedetect/MobilenetSSDFace/FDDBfacedet/FDDB-folds/fold-all.txt","/home/ljdong/PycharmProjects/facedetect/MobilenetSSDFace/FDDBfacedet"):
        img_result = detect(imgfile,True)

if __name__ == "__main__":
    main_show()

         