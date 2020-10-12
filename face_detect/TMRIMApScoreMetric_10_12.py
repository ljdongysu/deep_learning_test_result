from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx


def as_numpy(a):
    """Convert a (list of) mx.NDArray into numpy.ndarray"""
    if isinstance(a, (list, tuple)):
        out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
        try:
            out = np.concatenate(out, axis=0)
        except ValueError:
            out = np.array(out)
        return out
    elif isinstance(a, mx.nd.NDArray):
        a = a.asnumpy()
    return a

class TMRIMApScoreMetric(VOCMApMetric):
    """为了比赛输出得分"""
    def _update(self):
        """ update num_inst and sum_metric """

        aps = []
        recall, precs = self._recall_prec()

        #与原来代码有区别的位置，只能处理检测单类的结果
        l = 1
        rec = recall[1]
        prec = precs[1]

        ap = self._average_precision(rec, prec)
        aps.append(ap)
        if self.num is not None and l < (self.num - 1):
            self.sum_metric[l] = ap
            self.num_inst[l] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)

        plt.cla()
        #与原来代码有区别的位置
        l = 1
        rec = recall[1]
        prec = precs[1]
        pre_score_l = np.array(self._score[l])
        order = pre_score_l.argsort()[::-1]
        pre_score_l = pre_score_l[order]

        scores = rec * 0.4 + prec * 0.6
        plt.plot(rec, prec, '--', label=self.name[l]+" map")
        plt.plot(rec, scores, label=self.name[l]+" score")
        idx = np.argmax(scores)

        max_score = np.max(scores)
        plt.plot(rec[idx], max_score, 'ro')
        print('in {}| max score:{} prec:{} rec:{} threshold:{}'.format(
            self.name[l], max_score, prec[idx], rec[idx], pre_score_l[idx]))
        plt.legend()
        plt.savefig("precVSscore.jpg")
        # print("pre_score_l",pre_score_l)
        print("mps",aps)



if __name__ == "__main__":
    metric_map = TMRIMApScoreMetric(iou_thresh=0.5, class_names=["face"])
    metric_map.reset()
    # 人脸检测结果，文件格式：图像名“\n” 当前图像的人脸数量m “\n” 人脸框 "x,y,xmax,ymax,confidence\n" * m
    with open("/home/ljdong/data/wilderface/yufacedetectnet-open-v1.prototxt.txt","r") as f:
        test_result = f.read()

    test_result = test_result.split('\n')
    # 人脸标记结果，文件格式：图像名“\n” 当前图像的人脸数量n “\n” 人脸框 "x,y,xmax,ymax\n" * n 
    with open("/home/ljdong/data/wilderface/label_map.txt","r") as f:
        gt_result = f.read()
    gt_result = gt_result.split('\n')
    # print(test_result)
    num_r=0
    num_g=0
    while(num_r<len(test_result)-1 and num_g<len(gt_result)-1):
        num_r+=1
        num_g+=1
        test_boxes=[]
        test_confidence = []
        test_label = []
        gt_boxes = []
        gt_label = []
        test_faces = int(test_result[num_r])
        gt_faces = int(gt_result[num_g])
        #当人俩检测结果某一张图像人脸检测框数量为0的情况处理
        if(test_faces==0 or gt_faces==0):
            if(test_result[num_r-1]==gt_result[num_g-1]):
                num_r += test_faces+1
                num_g += gt_faces+1
                continue
            elif((test_faces==0 and gt_faces==0) and  test_result[num_r-1]!=gt_result[num_g-1]):
                print("groundtruth's index is different from test ressult")
                assert (0)
            elif(test_faces==0):
                num_r += test_faces+1
                num_g -=1
                continue
            elif(gt_faces==0):
                num_g += gt_faces+1
                num_r -=1
                continue
        for i in range(int(test_result[num_r])):
            num_r+=1
            test_boxes.append(list(map(float,test_result[num_r].split()[0:4])))
            test_confidence.append(test_result[num_r].split()[4])
            test_label.append(1)
        for j in range(int(gt_result[num_g])):
            num_g+=1
            gt_boxes.append(list(map(int,gt_result[num_g].split()[0:4])))
            gt_label.append(1)
        num_r+=1
        num_g+=1
        metric_map.update(np.array([test_boxes]),np.array([test_label]),np.array([test_confidence]),np.array([gt_boxes]),np.array([gt_label]))
    metric_map._update()
