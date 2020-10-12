from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx

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

        scores = rec * 0.6 + prec * 0.4
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

    def update(self, pred_bboxes, pred_labels, pred_scores,
               gt_bboxes, gt_labels, gt_difficults=None):
        iii = 0

        def bbox_iou(bbox_a, bbox_b, offset=0):
            """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

            Parameters
            ----------
            bbox_a : numpy.ndarray
                An ndarray with shape :math:`(N, 4)`.
            bbox_b : numpy.ndarray
                An ndarray with shape :math:`(M, 4)`.
            offset : float or int, default is 0
                The ``offset`` is used to control the whether the width(or height) is computed as
                (right - left + ``offset``).
                Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

            Returns
            -------
            numpy.ndarray
                An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
                bounding boxes in `bbox_a` and `bbox_b`.

            """
            if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
                raise IndexError("Bounding boxes axis 1 must have at least length 4")

            tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
            br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

            area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
            area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
            area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
            return area_i / (area_a[:, None] + area_b - area_i)
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

        if gt_difficults is None:
            gt_difficults = [None for _ in as_numpy(gt_labels)]

        for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores,
                                        gt_bboxes, gt_labels, gt_difficults]]):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]
            valid_gt = np.where(gt_label.flat >= 0)[0]
            gt_bbox = gt_bbox[valid_gt, :]
            gt_label = gt_label.flat[valid_gt].astype(int)
            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                self._n_pos[l] += np.logical_not(gt_difficult_l).sum()
                self._score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    self._match[l].extend((0,) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1

                iou = bbox_iou(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # print("iou: ",iou)
                # print("gt_index: ",gt_index)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            self._match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                self._match[l].append(1)
                            else:
                                self._match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        self._match[l].append(0)
                index_n=[]
                for index_,xx in enumerate(selec):
                    if(xx==False):
                        index_n.append(gt_boxes[index_])
                # print(index_n)
                return index_n

    def undetected_faces(self,img_name,face_box):
        writing_sring = img_name+"\n"
        writing_sring +=str(len(face_box))+'\n'
        for face in face_box:
            writing_sring+= str(face[0])+' '+str(face[1])+' '+str(face[2])+ ' '+ str(face[3])+'\n'
        return writing_sring

if __name__ == "__main__":
    #iou_thread 表示检测框与标记框的iou大于0.5才算检测到对应的人脸框
    metric_map = TMRIMApScoreMetric(iou_thresh=0.5, class_names=["face"])
    metric_map.reset()
    # 人脸检测结果，文件格式：图像名“\n” 当前图像的人脸数量m “\n” 人脸框 "x,y,xmax,ymax,confidence\n" * m
    with open("/home/ljdong/PycharmProjects/facedetect/MobilenetSSDFace/BMIfacedet/new_2/Bmi.txt","r") as f:
        test_result = f.read()

    test_result = test_result.split('\n')
    # 人脸标记结果，文件格式：图像名“\n” 当前图像的人脸数量n “\n” 人脸框 "x,y,xmax,ymax\n" * n
    with open("/home/ljdong/PycharmProjects/facedetect/MobilenetSSDFace/BMIfacedet/images/bmi_face_test_gt.txt","r") as f:
        gt_result = f.read()
    gt_result = gt_result.split('\n')
    # print(test_result)
    num_r=0
    num_g=0

    faces_un = []

    undetected_face = ''

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
        faces_un = metric_map.update(np.array([test_boxes]),np.array([test_label]),np.array([test_confidence]),np.array([gt_boxes]),np.array([gt_label]))
        if(len(faces_un)==0):
            continue
        undetected_face += metric_map.undetected_faces(gt_result[num_g-1-gt_faces-1],faces_un)
    with open("/home/ljdong/PycharmProjects/MobilenetSSDFace/BMIfacedet/images/bmi_face_undeteced.txt","w") as f:
        f.write(undetected_face)
    metric_map._update()
