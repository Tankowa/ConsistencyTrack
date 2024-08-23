import numpy as np
from collections import deque
import time
import torch
import torch.nn.functional as F 
import torchvision
from copy import deepcopy
from yolox.tracker import matching
from detectron2.structures import Boxes
from yolox.utils.box_ops import box_xyxy_to_cxcywh
from yolox.utils.boxes import xyxy2cxcywh
from torchvision.ops import box_iou,nms
from yolox.utils.cluster_nms import cluster_nms

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
#0.8 72.9
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self._tlwh=new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh=new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class ConsistencyTracker(object):
    def __init__(self,model,tensor_type,conf_thresh=0.25,det_thresh=0.7,nms_thresh_3d=0.6,nms_thresh_2d=0.7,interval=5,detections=None):

        self.frame_id = 0
        # BaseTrack._count=-1
        self.backbone=model.backbone
        self.feature_projs=model.projs
        self.consistency_model=model.head
        self.feature_extractor=self.consistency_model.head.box_pooler
        # self.det_thresh = det_thresh
        self.det_thresh = 0.8  #0.8 #0.85
        # self.association_thresh = conf_thresh
        self.association_thresh = 0.65  #0.65
        # self.low_det_thresh = 0.1
        # self.low_association_thresh = 0.2
        # self.nms_thresh_2d=nms_thresh_2d
        self.nms_thresh_2d=0.7 #0.7
        # self.nms_thresh_3d=nms_thresh_3d
        self.nms_thresh_3d=0.65 #0.65
        self.same_thresh=0.85 #0.9 #0.85
        self.pre_features=None
        self.data_type=tensor_type
        self.detections=detections

        #print(f'det_thresh:{self.det_thresh}, association_thresh:{self.association_thresh}, nms_thresh_3d:{self.nms_thresh_3d}, nms_thresh_2d:{self.nms_thresh_2d}')

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.max_time_lost = 30
        self.kalman_filter = KalmanFilter()

        self.repeat_times=1   #8
        self.dynamic_time=True
        
        self.sampling_steps=1 #1
        self.num_boxes=500 #2000
        #采样时间步
        self.track_t=39
        self.mot17=True

        self.track_thresh = 0.2 #0.2


    def update(self,cur_image):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        un_detections = []
        cur_features,mate_info=self.extract_feature(cur_image=cur_image)
        mate_shape,mate_device,mate_dtype=mate_info
        self.consistency_model.device=mate_device
        self.consistency_model.dtype=mate_dtype
        b,_,h,w=mate_shape
        images_whwh=torch.tensor([w, h, w, h], dtype=mate_dtype, device=mate_device)[None,:].expand(4*b, 4)
        if self.frame_id==1:
            #print(1)
            if self.pre_features is None:
                self.pre_features=cur_features
            inps=self.prepare_input(self.pre_features,cur_features)
            consistency_outputs,conf_scores,association_time=self.consistency_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                dynamic_time=self.dynamic_time,track_candidate=self.repeat_times)
            _,_,detections=self.consistency_postprocess(consistency_outputs,conf_scores,conf_thre=self.association_thresh,nms_thre=self.nms_thresh_3d)
            detections=self.consistency_det_filt(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)
            # detections=np.array(self.detections[self.frame_id])
            # detections=detections[detections[:,5]>self.det_thresh]
            for det in detections:
                track=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                track.activate(self.kalman_filter, self.frame_id)
                self.tracked_stracks.append(track)
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]
            return output_stracks,association_time
        else:

            '''判断上一帧的正在跟踪的轨迹信息，是否已经被激活'''
            #没有被激活的轨迹
            unconfirmed = []
            #被激活的轨迹
            tracked_stracks = []  # type: list[STrack]
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)

            #print("unconfirmed",unconfirmed)


            '''根据上一祯，初始化轨迹框信息'''
            ref_bboxes=[STrack.tlwh_to_tlbr(track._tlwh) for track in tracked_stracks]
            ref_bboxes_1 = [STrack.tlwh_to_tlbr(track._tlwh) for track in self.tracked_stracks]


            STrack.multi_predict(self.tracked_stracks)
            #print("tracked_stracks:",tracked_stracks)
            #print("self.tracked_stracks", self.tracked_stracks)
            #ref_bboxes_2 = [STrack.tlwh_to_tlbr(track._tlwh) for track in tracked_stracks]

            '''带入head计算，过score_thresh,nms'''
            inps=self.prepare_input(self.pre_features,cur_features)

            if len(ref_bboxes_1)>0:
                bboxes=box_xyxy_to_cxcywh(torch.tensor(np.array(ref_bboxes_1))).type(self.data_type).reshape(1,-1,4).repeat(2,1,1)
            else:
                bboxes=None
            #print(1)
            # ref_num_proposals=self.proposal_schedule(len(ref_bboxes))
            # ref_sampling_steps=self.sampling_steps_schedule(len(ref_bboxes))
            '''推理过程'''
            consistency_outputs,conf_scores,association_time=self.consistency_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                ref_targets=bboxes,dynamic_time=self.dynamic_time,track_candidate=self.repeat_times,consistency_t=self.track_t)

            #算法7-17行
            #score取值范围[0.65,1]*[0,1],前者为关联分数,后者为检测的准确score
            consistency_ref_detections,consistency_track_detections,detections=self.consistency_postprocess(consistency_outputs,
                                                                                                      conf_scores,
                                                                                                      conf_thre=self.association_thresh,
                                                                                                      nms_thre=self.nms_thresh_3d)
            #不筛框的置信度
            detections=self.consistency_det_filt_second(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)

            '''根据track_thresh筛选高置信度的框和低置信度的框，detections是带入模型的第二个batchsize'''
            scores = detections[:, 5]
            #print("scores:",scores)
            remain_inds = scores > self.track_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.track_thresh
            inds_second = np.logical_and(inds_low, inds_high)  # 筛选分数处于0.1<分数<阈值的
            #筛选出置信度分数较小的
            detections_second = detections[inds_second]
            #筛选出置信分数大的
            detections = detections[remain_inds]



            # detections=np.array(self.detections[self.frame_id])
            # if len(detections)>0:
            #     detections=detections[detections[:,5]>self.det_thresh]
            consistency_ref_detections,consistency_track_detections=self.consistency_track_filt_second(consistency_ref_detections,
                                                                                          consistency_track_detections,
                                                                                          conf_thre=self.det_thresh,
                                                                                          nms_thre=self.nms_thresh_2d)
            consistency_ref_detections_1 = consistency_ref_detections.copy()
            consistency_track_detections_1 = consistency_track_detections.copy()
            '''根据track_thresh筛选高置信度的框和低置信度的框，带入模型的第二个batchsize，分为前一帧的检测框和后一帧的检测框'''
            # 获取置信度分数列
            scores_ref = consistency_ref_detections[:, 5]
            scores_cur = consistency_track_detections[:, 5]

            # 筛选出同时大于阈值的索引
            inds_ref = scores_ref > self.track_thresh
            inds_cur = scores_cur > self.track_thresh

            # 取两个检测结果中同时满足条件的索引
            inds_both = np.logical_and(inds_ref, inds_cur)
            #print("inds_both",inds_both)
            # 筛选出符合条件的检测结果
            consistency_ref_detections = consistency_ref_detections[inds_both]
            consistency_track_detections = consistency_track_detections[inds_both]
            #print("consistency_ref_detections",consistency_ref_detections)
            #print("consistency_track_detections", consistency_track_detections)

            scores_ref_1 = consistency_ref_detections_1[:, 5]
            scores_cur_1 = consistency_track_detections_1[:, 5]
            # 筛选出置信分数同时小于0.1的检测结果s
            second_inds_ref = scores_ref_1 < 0.1
            second_inds_cur = scores_cur_1 < 0.1
            second_inds_both = np.logical_and(second_inds_ref, second_inds_cur)

            inds_ref_t = np.logical_not(inds_both,second_inds_both)
            #print("inds_ref_t",inds_ref_t)
            consistency_ref_detections_second = consistency_ref_detections_1[inds_ref_t]
            consistency_track_detections_second = consistency_track_detections_1[inds_ref_t]

            #print("consistency_ref_detections_second",consistency_ref_detections_second)
            #print("consistency_track_detections_second", consistency_track_detections_second)

            start_time=time.time()
            '''预测更新了上一帧的轨迹信息，变为当前帧通过卡尔曼滤波预测的结果'''
            #STrack.multi_predict(self.tracked_stracks)

            '''第一次匹配，上一帧轨迹信息转换出的轨迹框与高置信度的前一帧的检测框进行匹配'''

            """       
            dists = matching.iou_distance(ref_bboxes_2, consistency_track_detections[:,:4])
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)
            """
            dists = matching.iou_distance(ref_bboxes, consistency_ref_detections[:,:4])
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)

            op = False

            if len(matches)>0:
                op=True
                # fix position with detection result
                '''第二次小匹配，高置信度的当前帧的检测框进行匹配与第二个batchsize的检测框进行匹配'''
                dists_fix=matching.iou_distance(consistency_track_detections[matches[:,1],:4],detections[:,:4])
                matches_fix, u_track_fix, u_detection_fix = matching.linear_assignment(dists_fix, thresh=self.same_thresh)
                if len(matches_fix)>0:
                    #print(1)
                    consistency_track_detections[matches[:,1]][matches_fix[:,0],:4]=detections[matches_fix[:,1],:4]
                '''更新该detections为没有匹配上的部分'''
                # filt detection with tracked result
                un_detections=detections[u_detection_fix]

            '''第一次更新轨迹信息，第一次匹配成功的轨迹和检测框，从预测的当前帧的轨迹列表中选取匹配上的轨迹，用consistency_track_detections中的信息去更新'''
            ref_box_t=[]
            track_box_t=[]
            for itracked, idet in matches:
                track = tracked_stracks[itracked]  #maybe应该删除
                ref_box_t.append(STrack.tlwh_to_tlbr(track._tlwh))
                det = consistency_track_detections[idet]
                track_box_t.append(det[:4])
                new_strack=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                if track.state == TrackState.Tracked:
                    #print(0)
                    track.update(new_strack, self.frame_id)
                    activated_starcks.append(track)
                else:
                    #new_id标为false，不是初始化新轨迹
                    #print(1)
                    track.re_activate(new_strack, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            #计算下次操作的采样时间步
            if len(ref_box_t)>0:
                self.track_t=self.extract_mean_track_t(np.array(ref_box_t),np.array(track_box_t))
                #print(self.track_t)

            """将上一帧标记为lost的轨迹，重新预测到这一针的位置"""
            STrack.multi_predict(self.lost_stracks)
            u_track_lost=[]
            '''第三次匹配，lost轨迹与Unmatched_detections匹配'''

            #print("the first:",self.lost_stracks)
            if(op):
                dists_lost = matching.iou_distance([track.tlbr for track in self.lost_stracks], un_detections[:4])
                matches_lost, u_track_lost, u_detection_lost = matching.linear_assignment(dists_lost, thresh=self.same_thresh)

                """匹配上的激活更新"""
                for itracked, idet in matches_lost:
                    track = self.lost_stracks[itracked]
                    det = un_detections[idet]
                    new_strack=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                    if track.state == TrackState.Tracked:
                        track.update(new_strack, self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(new_strack, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                #print("the second:",self.lost_stracks)
                """未匹配上的轨迹丢弃"""

                for it in u_track_lost:
                    '''对才时间lost的轨迹进行丢弃'''
                    track = self.lost_stracks[it]
                    if self.frame_id - track.end_frame > self.max_time_lost:
                        track.mark_removed()
                        removed_stracks.append(track)

                """未匹配上的检测框，初始化轨迹"""
                for inew in u_detection_lost:
                    det = un_detections[inew]
                    track = STrack(STrack.tlbr_to_tlwh(det[:4]),det[5])
                    if track.score < 0.6: #0.
                        continue
                    # 只有第一帧新建的轨迹会被标记为is_activate=True，其他帧不会
                    track.activate(self.kalman_filter, self.frame_id)
                    # 把新的轨迹加入到当前活跃轨迹中
                    activated_starcks.append(track)


            #print("consistency_ref_detections_second",consistency_ref_detections_second)
            '''第二次匹配，低置信度检测框的匹配过程'''
            #ref_bboxes = [STrack.tlwh_to_tlbr(track._tlwh) for track in tracked_stracks]
            #r_tracked_stracks = [tracked_stracks[i] for i in u_track if tracked_stracks[i].state == TrackState.Tracked]
            r_tracked_stracks = [tracked_stracks[i] for i in u_track if tracked_stracks[i].state == TrackState.Tracked]

            """
            if len(u_track_lost):
                w = [self.lost_stracks[i] for i in u_track_lost if self.lost_stracks[i].state == TrackState.Tracked]
                #print(w)
                r_tracked_stracks = joint_stracks(r_tracked_stracks,w)
            """
            #print("r_tracked_stracks",r_tracked_stracks)
            ref_bboxes_t = [STrack.tlwh_to_tlbr(track._tlwh) for track in r_tracked_stracks]
            #print("ref_bboxes_t",ref_bboxes_t)
            # 计算r_tracked_stracks和detections_second的iou_distance(代价矩阵)
            dists = matching.iou_distance(ref_bboxes_t, consistency_ref_detections_second)
            #print("dists",dists)
            # 用match_thresh = 0.8过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
            matches, u_track, u_detection_second = matching.linear_assignment(dists,
                                                                              thresh=0.5)  # 分数比较低的目标框没有匹配到轨迹就会直接被扔掉，不会创建新的轨迹
            #print("U_track:",u_track)

            if len(matches)>0:
                # fix position with detection result
                dists_fix=matching.iou_distance(consistency_track_detections_second[matches[:,1],:4],detections_second[:,:4])
                matches_fix, u_track_fix, u_detection_fix = matching.linear_assignment(dists_fix, thresh=0.5)
                if len(matches_fix)>0:
                    consistency_track_detections_second[matches[:,1]][matches_fix[:,0],:4]=detections_second[matches_fix[:,1],:4]

            #print(r_tracked_stracks[0]._tlwh)

            #STrack.multi_predict(r_tracked_stracks)
            #print(r_tracked_stracks)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                #track.predict()
                #ref_box_t.append(STrack.tlwh_to_tlbr(track._tlwh))
                det = consistency_track_detections_second[idet]
                #track_box_t.append(det[:4])
                new_strack=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                if track.state == TrackState.Tracked:
                    track.update(new_strack, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(new_strack, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            #STrack.multi_predict(r_tracked_stracks)

            #应该用卡尔曼滤波去更新其轨迹然后存入
            for it in u_track:
                #print(track._tlwh)
                track = r_tracked_stracks[it]
                #track.predict()
                #print(track._tlwh)
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)


            un_bboxes = [STrack.tlwh_to_tlbr(track._tlwh) for track in unconfirmed]

            '''第四次匹配，没有激活的轨迹与第一次匹配未匹配成功的检测框之间匹配'''
            detections = [consistency_track_detections[i] for i in u_detection]
            # 计算unconfirmed和detections的iou_distance(代价矩阵)
            # unconfirmed是不活跃的轨迹（过了30帧）
            dists = matching.iou_distance(un_bboxes, detections)
            #if not self.args.mot20:
            #    dists = matching.fuse_score(dists, detections)
            # 用match_thresh = 0.8过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
            # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
            '''匹配上的激活'''
            for itracked, idet in matches:
                track = unconfirmed[itracked]
                det = detections[idet]
                new_strack=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                if track.state == TrackState.Tracked:
                    track.update(new_strack, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(new_strack, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            # 遍历u_unconfirmed，调用mark_removd方法，并加入removed_stracks
            '''未匹配上轨迹移除'''
            for it in u_unconfirmed:
                # 中途出现一次的轨迹和当前目标框匹配失败，删除该轨迹（认为是检测器误判）
                # 真的需要直接删除吗？？
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            '''未匹配上的检测况初始为新的轨迹'''
            for inew in u_detection:
            # for inew in range(len(detections)):
                det = detections[inew]
                track=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                #if track.score < 0.6:
                #    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)


            '''对才时间lost的轨迹进行丢弃'''
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
            

            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
            # get scores of lost tracks
           

        self.pre_features=cur_features
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks,association_time+time.time()-start_time
    
    def extract_feature(self,cur_image):
        fpn_outs=self.backbone(cur_image)
        cur_features=[]
        for proj,l_feat in zip(self.feature_projs,fpn_outs):
            cur_features.append(proj(l_feat))
        mate_info=(cur_image.shape,cur_image.device,cur_image.dtype)
        return cur_features,mate_info

    def extract_mean_track_t(self,pre_box,cur_box):
        # "xyxy"
        pre_box=xyxy2cxcywh(pre_box)
        cur_box=xyxy2cxcywh(cur_box)
        abs_box=np.abs(pre_box-cur_box)
        abs_percent=np.sum(abs_box/(pre_box+1e-5),axis=1)/4
        #print(abs_percent)
        track_t=np.mean(abs_percent)
        #print(track_t)
        # min(max(int(track_t*40),1),39)
        # min(max(int((np.exp(track_t)-1)/(np.exp(0)-1)*40),1),39)
        # min(max(int(np.log(track_t+1)/np.log(2)*40),1),39)
        return min(max(int(np.log(track_t+1)/np.log(2)*40),1),39)

    def custom_transform(self,x):
        # 对输入值进行拉伸和缩放
        x = torch.log(x + 1e-9)  # 使用对数函数
        x = x / torch.log(torch.tensor(1.01))  # 缩放
        x = x - torch.min(x)  # 移动到0
        x = x / torch.max(x)  # 归一化到0到1范围
        x = x * 0.9 + 0.1  # 缩放到0.1到1范围'''
        return x
    



    def consistency_postprocess(self,consistency_outputs,conf_scores,nms_thre=0.7,conf_thre=0.6):

        pre_prediction,cur_prediction=consistency_outputs.split(len(consistency_outputs)//2,dim=0)

        output = [None for _ in range(len(pre_prediction))]
        for i,(pre_image_pred,cur_image_pred,association_score) in enumerate(zip(pre_prediction,cur_prediction,conf_scores)):
            #print("pre_image_pred",pre_image_pred)
            #print("cur_image_pred",cur_image_pred)
            #print("pre_image_pred",pre_image_pred[:,:4])
            association_score=association_score.flatten()
            # If none are remaining => process next image
            if not pre_image_pred.size(0):
                continue
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections=torch.zeros((2,len(cur_image_pred),7),dtype=cur_image_pred.dtype,device=cur_image_pred.device)

            detections[0,:,:4]=pre_image_pred[:,:4]
            detections[1,:,:4]=cur_image_pred[:,:4]
            detections[0,:,4]=association_score
            detections[1,:,4]=association_score
            #print("association_score",association_score)
            detections[0,:,5]=self.custom_transform(torch.sqrt(torch.sigmoid(pre_image_pred[:,4]/10)))
            #print("detections[0,:,5]",detections[0,:,5])
            #print("t:",self.custom_transform_two(torch.sqrt(torch.sigmoid(pre_image_pred[:,4]/10))))
            #detections[0,:,5]=torch.sqrt(torch.sigmoid(pre_image_pred[:,4])*association_score)
            #print("detections[0,:,5]",detections[0,:,5])
            detections[1,:,5]=self.custom_transform(torch.sqrt(torch.sigmoid(cur_image_pred[:,4]/10)))
            #detections[0, :, 6] = pre_image_pred[:,4]
            #detections[1, :, 6] = cur_image_pred[:,4]

            score_out_index=association_score>conf_thre
            detections=detections[:,score_out_index,:]

            if not detections.size(1):
                output[i]=detections
                continue

            nms_out_index_3d = cluster_nms(
                                        detections[0,:,:4],
                                        detections[1,:,:4],
                                        # value[score_out_index],
                                        detections[0,:,4],
                                        iou_threshold=nms_thre)

            detections = detections[:,nms_out_index_3d,:]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output[0][0],output[0][1],torch.cat([output[1][0],output[1][1]],dim=0) if len(output)>=2 else None

    def consistency_track_filt(self, ref_detections, track_detections, conf_thre=0.6, nms_thre=0.7):

        if not ref_detections.size(1):
            return ref_detections.cpu().numpy(), track_detections.cpu().numpy()

        scores = ref_detections[:, 5]
        score_out_index = scores > conf_thre
        ref_detections = ref_detections[score_out_index]
        track_detections = track_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
            ref_detections[:, :4],
            ref_detections[:, 5],
            ref_detections[:, 6],
            nms_thre,
        )
        return ref_detections[nms_out_index].cpu().numpy(), track_detections[nms_out_index].cpu().numpy()

    def consistency_det_filt_second(self, consistency_detections, conf_thre=0.6, nms_thre=0.7):

        if not consistency_detections.size(1):
            return consistency_detections.cpu().numpy()

        #scores = consistency_detections[:, 5]
        #score_out_index = scores > conf_thre
        #consistency_detections = consistency_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
            consistency_detections[:, :4],
            consistency_detections[:, 5],
            consistency_detections[:, 6],
            nms_thre,
        )
        return consistency_detections[nms_out_index].cpu().numpy()


    def consistency_track_filt_second(self,ref_detections,track_detections,conf_thre=0.6,nms_thre=0.7):

        if not ref_detections.size(1):
            return ref_detections.cpu().numpy(),track_detections.cpu().numpy()
        
        #scores=ref_detections[:,5]
        #score_out_index=scores>conf_thre
        #ref_detections=ref_detections[score_out_index]
        #track_detections=track_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
                ref_detections[:, :4],
                ref_detections[:, 5],
                ref_detections[:, 6],
                nms_thre,
            )
        return ref_detections[nms_out_index].cpu().numpy(),track_detections[nms_out_index].cpu().numpy()

    def consistency_det_filt(self,consistency_detections,conf_thre=0.6,nms_thre=0.7):

        if not consistency_detections.size(1):
            return consistency_detections.cpu().numpy()

        scores=consistency_detections[:,5]
        score_out_index=scores>conf_thre
        consistency_detections=consistency_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
                consistency_detections[:, :4],
                consistency_detections[:, 5],
                consistency_detections[:, 6],
                nms_thre,
            )
        return consistency_detections[nms_out_index].cpu().numpy()
    
    def proposal_schedule(self,num_ref_bboxes):
        # simple strategy
        return 16*num_ref_bboxes
    
    def sampling_steps_schedule(self,num_ref_bboxes):
        min_sampling_steps=1
        max_sampling_steps=4
        min_num_bboxes=10
        max_num_bboxes=100
        ref_sampling_steps=(num_ref_bboxes-min_num_bboxes)*(max_sampling_steps-min_sampling_steps)/(max_num_bboxes-min_num_bboxes)+min_sampling_steps

        return min(max(int(ref_sampling_steps),min_sampling_steps),max_sampling_steps)

    def vote_to_remove_candidate(self,track_ids,detections,vote_iou_thres=0.75,sorted=False,descending=False):

        box_pred_per_image, scores_per_image=detections[:,:4],detections[:,4]*detections[:,5]
        score_track_indices=torch.argsort((track_ids+scores_per_image),descending=True)
        track_ids=track_ids[score_track_indices]
        scores_per_image=scores_per_image[score_track_indices]
        box_pred_per_image=box_pred_per_image[score_track_indices]

        assert len(track_ids)==box_pred_per_image.shape[0]

        # vote guarantee only one track id in track candidates
        keep_mask = torch.zeros_like(scores_per_image, dtype=torch.bool)
        for class_id in torch.unique(track_ids):
            curr_indices = torch.where(track_ids == class_id)[0]
            curr_keep_indices = nms(box_pred_per_image[curr_indices],scores_per_image[curr_indices],vote_iou_thres)
            candidate_iou_indices=box_iou(box_pred_per_image[curr_indices],box_pred_per_image[curr_indices])>vote_iou_thres
            counter=[]
            for cluster_indice in candidate_iou_indices[curr_keep_indices]:
                cluster_scores=scores_per_image[curr_indices][cluster_indice]
                counter.append(len(cluster_scores)+torch.mean(cluster_scores))
            max_indice=torch.argmax(torch.tensor(counter).type(self.data_type))
            keep_mask[curr_indices[curr_keep_indices][max_indice]] = True
        
        keep_indices = torch.where(keep_mask)[0]        
        track_ids=track_ids[keep_indices]
        box_pred_per_image=box_pred_per_image[keep_indices]
        scores_per_image=scores_per_image[keep_indices]

        if sorted and not descending:
            descending_indices=torch.argsort(track_ids)
            track_ids=track_ids[descending_indices]
            box_pred_per_image=box_pred_per_image[descending_indices]
            scores_per_image=scores_per_image[descending_indices]

        return track_ids.cpu().numpy(),box_pred_per_image.cpu().numpy(),scores_per_image.cpu().numpy()

    def prepare_input(self,pre_features,cur_features):
        inps_pre_features=[]
        inps_cur_Features=[]
        for l_pre_feat,l_cur_feat in zip(pre_features,cur_features):
            inps_pre_features.append(torch.cat([l_pre_feat.clone(),l_cur_feat.clone()],dim=0))
            inps_cur_Features.append(torch.cat([l_cur_feat.clone(),l_cur_feat.clone()],dim=0))
        return (inps_pre_features,inps_cur_Features)

    # def get_targets_from_tracklet_db(self):
    #     ref_mask=self.tracklet_db[:,-1,:5].sum(-1)>0
    #     ref_bbox=deepcopy(self.tracklet_db[ref_mask,-1,:4])
    #     ref_track_ids=np.arange(len(self.tracklet_db))[ref_mask]
    #     return ref_bbox,ref_track_ids


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

from sklearn.metrics.pairwise import cosine_similarity
def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    # if len(stracksa)>0 and len(stracksb)>0:
    #     # fix a derection bug
    #     pcosdist=cosine_similarity(
    #         [track.mean[4:6] for track in stracksa],
    #         [track.mean[4:6] for track in stracksb])
    #     pdist=(pdist+pcosdist)/2
    
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if stracksa[p].mean is not None and stracksb[q].mean is not None:
            x,y=stracksa[p].mean[4:6],stracksa[p].mean[4:6]
            cosine_dist=1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-06)
            if cosine_dist>0.15:
                continue
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


