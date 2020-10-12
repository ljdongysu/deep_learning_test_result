# 测试MobileNetSSD检测结果

import numpy as np  
import sys,os  
import cv2
#caffe_root = '/media/ryan/F/Projects/caffe/'
#sys.path.insert(0, caffe_root + 'python')  
import caffe  

net_file= '/home/ljdong/Software/Hi3531DV200R001C02SPC003/SVP_PC/HiSVP_PC_V1.3.0.7/software/data/detection/mobilenetssd/model/face_deploy_8.prototxt'  
caffe_model='/home/ljdong/Software/Hi3531DV200R001C02SPC003/SVP_PC/HiSVP_PC_V1.3.0.7/software/data/detection/mobilenetssd/model/60000_8_focal_g1_bgs_test.caffemodel'

test_dir = "/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/rgb_image"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background','face')


def preprocess(src):
    img = cv2.resize(src, (256,256))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img[:,:,0] = img[:,:,0] #- 127.5
    img[:,:,1] = img[:,:,1] #- 127.5
    img[:,:,2] = img[:,:,2]# - 127.5
    img = img * 0.003921
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    print("out.shape: ",out['detection_out'][0,0,:,2:])
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img  = preprocess(origimg)
    
    img = img.astype(np.float16)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)
    print("imagefile: ",imgfile)
    for i in range(len(box)):
       if conf[i] < 0.2:
           continue
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       print("(",box[i][0],",", box[i][1],")",",","(",box[i][2],",", box[i][3],")")
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
    #    title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       title = "."

       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
    # cv2.imwrite("/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/test3_result/" + imgfile.split('/')[-1],origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
