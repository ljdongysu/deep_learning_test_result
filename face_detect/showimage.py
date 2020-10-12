import cv2
def show_simgle_img():
    image = cv2.imread("/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/two_persons_face_256x256.jpg")

    cv2.rectangle(image,(54  ,12 ),( 90 ,74 ),(0,255,255),1) 
    cv2.rectangle(image,(168 ,66 ),( 204 ,121 ),(0,255,255),1) 

    # 138 166 169 201 1.000000
    # [level]:Info,[func]:sample_common_svp_nnie_print_sw_detect_result [line]:763 [info]:169 54 199 93 1.000000
    # [level]:Info,[func]:sample_common_svp_nnie_print_sw_detect_result [line]:763 [info]:129 44 174 101 0.990479

    # cv2.rectangle(image,(58 ,230 ),( 89 ,265 ),(0,255,255),5)    
    # cv2.rectangle(image,(9 ,86 ),( 39 ,126 ),(0,255,255),5)    
    cv2.imshow("1",image)
    cv2.waitKey(0)
def show_image_result(img_file1,img_file2):
    img_num1 = 0
    img_num2 = 0
    while(1):
        img1 = img_file1[img_num1].split(' ')[0]
        rectangle_num1 = int(img_file1[img_num1].split(' ')[1])

        img2 = img_file2[img_num2].split(' ')[0]
        rectangle_num2 = int(img_file2[img_num2].split(' ')[1])
        if (img1.split('/')[-1] != img2.split('/')[-1]):
            print(img1,img2,img_num1,img_num2)
            print("result_file error!\n")
            assert 0
        else:
            print(img1,img2)
        image1 = "/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/rgb_image_256/" +  img1.split('/')[-1].split('.')[0] + '.jpg'
        image2 = "/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/rgb_image/" +  img1.split('/')[-1].split('.')[0] + '.jpg'

        
        image_show = cv2.imread(image1)
        h0,w0,c0 = image_show.shape
        image_shape = cv2.imread(image2)
        h,w,c = image_shape.shape
        print(image_shape.shape)

        for _ in range(rectangle_num1):
            img_num1 += 1
            x_min = int(img_file1[img_num1].split(' ')[0])
            y_min = int(img_file1[img_num1].split(' ')[1])
            x_max = int(img_file1[img_num1].split(' ')[2])
            y_max = int(img_file1[img_num1].split(' ')[3])
            score = img_file1[img_num1].split(' ')[4]
            cv2.rectangle(image_show,(x_min ,y_min ),( x_max ,y_max ),(0,255,255),1) 
            cv2.putText(image_show, "mssd:%s"%score, (x_min,y_min), cv2.FONT_ITALIC, 0.6, (0, 255,255),1)

        for _ in range(rectangle_num2):
            img_num2 += 1
            x_min = int((float)(img_file2[img_num2].split(' ')[0])/w * w0)
            y_min = int(float(img_file2[img_num2].split(' ')[1])/h * h0)
            x_max = int(float(img_file2[img_num2].split(' ')[2])/w * w0)
            y_max = int(float(img_file2[img_num2].split(' ')[3])/h * h0)
            score = img_file2[img_num2].split(' ')[4]
            cv2.rectangle(image_show,(x_min ,y_min ),( x_max ,y_max ),(0,0,255),1) 
            cv2.putText(image_show, "yunet:%s"%score, (x_min,y_min), cv2.FONT_ITALIC, 0.6, (0, 0,255), 1)

        cv2.imshow("image-show",image_show)
        cv2.waitKey(0)
        if img_num1 == len(img_file1) -1 and img_num2 == len(img_file2) -1:
            break
        img_num1 += 1
        img_num2 += 1
def show_image_result_2(img_file1,img_file2, root_dir):
    img_num1 = 0
    img_num2 = 0
    while(img_num1 < len(img_file1) and img_num2 < len(img_file2)):
        img1 = img_file1[img_num1]
        img_num1 += 1
        print(img_file1)
        print(img_num1,img_file1[img_num1])
        rectangle_num1 = int(img_file1[img_num1])

        img2 = img_file2[img_num2]
        img_num2 += 1
        rectangle_num2 = int(img_file2[img_num2])
        if (img1.split('/')[-1] != img2.split('/')[-1]):
            print(img1,img2,img_num1,img_num2)
            print("result_file error!\n")
            assert 0
        else:
            print(img1,img2)
        image1 = '/'.join([root_dir,img1])
        image2 = '/'.join([root_dir,img2])
        print(root_dir,image1)
        image_show = cv2.imread(image1)
        print(image_show)
        h0,w0,c0 = image_show.shape
        image_shape = cv2.imread(image2)
        h,w,c = image_shape.shape
        print(image_shape.shape)

        for _ in range(rectangle_num1):
            img_num1 += 1
            x_min = int(img_file1[img_num1].split(' ')[0])
            y_min = int(img_file1[img_num1].split(' ')[1])
            x_max = int(img_file1[img_num1].split(' ')[2])
            y_max = int(img_file1[img_num1].split(' ')[3])
            score = img_file1[img_num1].split(' ')[4]
            cv2.rectangle(image_show,(x_min ,y_min ),( x_max ,y_max ),(0,255,255),1) 
            cv2.putText(image_show, "mssd:%s"%score, (x_min,y_min), cv2.FONT_ITALIC, 0.6, (0, 255,255),1)

        for _ in range(rectangle_num2):
            img_num2 += 1
            x_min = int((float)(img_file2[img_num2].split(' ')[0])/w * w0)
            y_min = int(float(img_file2[img_num2].split(' ')[1])/h * h0)
            x_max = int(float(img_file2[img_num2].split(' ')[2])/w * w0)
            y_max = int(float(img_file2[img_num2].split(' ')[3])/h * h0)
            score = img_file2[img_num2].split(' ')[4]
            cv2.rectangle(image_show,(x_min ,y_min ),( x_max ,y_max ),(0,0,255),1) 
            cv2.putText(image_show, "yunet:%s"%score, (x_min,y_min), cv2.FONT_ITALIC, 0.6, (0, 0,255), 1)

        cv2.imshow("image-show",image_show)
        cv2.waitKey(0)
        if img_num1 == len(img_file1) -1 and img_num2 == len(img_file2) -1:
            break
        img_num1 += 1
        img_num2 += 1



  
def main():
    # show_simgle_img()
    f1 = open("/home/ljdong/data/wilderface/face_deploy_8.prototxt.txt",'r')
    f2 = open("/home/ljdong/data/wilderface/yufacedetectnet-open-v1.prototxt.txt",'r')
    root_dir = "/home/ljdong/data/wilderface/images"
    file_list1 = f1.read().split('\n')
    file_list2 = f2.read().split('\n')
    show_image_result_2(file_list1,file_list2, root_dir)
    f1.close()
    f2.close()

if __name__ == "__main__":
    main()