import os


# 遍历文件夹
def walkFile(file,filename):
    wf = open(filename,'w')
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            print(os.path.join(root, f))
            wf.write(os.path.join(root, f))
            wf.write("\n")
        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))


def main():

    walkFile("/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/bgr_img/","/home/ljdong/code/qt/build-jpeg2bgr-Desktop_Qt_5_14_0_GCC_64bit-Debug/filename.txt")


if __name__ == '__main__':
    main()
