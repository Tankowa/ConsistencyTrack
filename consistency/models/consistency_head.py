import math
import random
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms, box_iou

from .consistency_losses import SetCriterionDynamicK, HungarianMatcherDynamicK
from .consistency_models import DynamicHead

from yolox.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from yolox.utils import synchronize
from detectron2.layers import batched_nms
import time

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def get_sigmas_karras(sigma_min, sigma_max, rho, n_steps):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class consistencyHead(nn.Module):
    """
    Implement consistencyHead
    """

    def __init__(self,
                 num_classes,
                 width=1.0,
                 strides=[8, 16, 32],
                 num_proposals=500,
                 num_heads=3, ):
        super().__init__()
        self.device = "cpu"
        self.dtype = torch.float32
        self.width = width
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        # self.num_proposals = 512
        self.hidden_dim = int(256 * width)
        self.num_heads = num_heads

        # build consistency
        timesteps = 1000
        sampling_timesteps = 1
        # self.objective = 'pred_x0'：这个变量指定了扩散过程的目标。在这里，目标是预测某个初始状态（x0）。
        self.objective = 'pred_x0'

        self.num_timesteps = int(timesteps)
        self.distillation =True
        # tracking setting
        self.inference_time_range = 1
        self.track_candidate = 1
        self.candidate_num_strategy = max
        # 采样时间步数通常用于确定在扩散过程中对模型状态进行多少次采样
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.box_renewal = True
        self.use_ensemble = True

        # Build consistency parameters
        self.sigma_max = 40
        self.sigma_min = 0.002
        self.sigma_data = 0.5
        self.rho = 7
        self.n_steps = 40
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sigmas = get_sigmas_karras(self.sigma_min, self.sigma_max, self.rho, self.n_steps)
        #print(self.sigmas)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for consistency q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the consistency chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # Build Dynamic Head.

        class_weight = 2.0
        giou_weight = 2.0
        l1_weight = 5.0
        no_object_weight = 0.1
        self.deep_supervision = True
        self.use_focal = True
        self.use_fed_loss = False
        self.use_nms = False
        self.pooler_resolution = 7
        self.noise_strategy = "xywh"

        self.head = DynamicHead(num_classes, self.hidden_dim, self.pooler_resolution, strides,
                                [self.hidden_dim] * len(strides), return_intermediate=self.deep_supervision,
                                num_heads=self.num_heads, use_focal=self.use_focal, use_fed_loss=self.use_fed_loss)
        # Loss parameters:

        # Build Criterion.
        # 匹配算法匈牙利
        matcher = HungarianMatcherDynamicK(
            cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal,
            use_fed_loss=self.use_fed_loss
        )

        # 不同的num_head有不同的头
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal, use_fed_loss=self.use_fed_loss)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_consistency_predictions(self, backbone_feats, images_whwh, boxes_test, sig, time_cond, lost_features=None, fix_bboxes=False):

        start_time = time.time()
        bs = len(boxes_test) // 2
        outputs_class, outputs_coord, outputs_score = self.consistency_function(self.head, backbone_feats, boxes_test, sig, time_cond,
                                                                 images_whwh, lost_features, fix_bboxes)

        end_time = time.time()
        # outputs_coord = torch.clamp(outputs_coord, min=0, max=255)
        # x_start xyxy255
        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        # x_start xyxy01
        x_start = x_start / images_whwh[:, None, :]

        # x_start_xyxy01 = x_start.clone()
        # x_start cxcy01
        x_start = box_xyxy_to_cxcywh(x_start)

        # x_start cxcy[-self.scale, self.scale]
        x_start = (x_start * 2 - 1.)
        # x_start cxcy[-self.scale, self.scale]。
        x_start = torch.clamp(x_start, min=-1, max=1)
        # pred_noise = self.predict_noise_from_start(x, t, x_start)
        # x_start cxcy[-self.scale, self.scale],
        return boxes_test, x_start, outputs_class, outputs_coord, outputs_score, end_time-start_time
    @torch.no_grad()  # 2*b,4results = self.new_ddim_sample(features,images_whwh,targets,dynamic_time=False)#new_ddim_sample
    def new_ddim_sample(self, backbone_feats, images_whwh, ref_targets=None, dynamic_time=True, num_timesteps=1,
                        num_proposals=500,
                        inference_time_range=1, track_candidate=1, consistency_t=40, clip_denoised=True):
        batch = images_whwh.shape[0] // 2
        self.sampling_timesteps, self.num_proposals, self.track_candidate, self.inference_time_range = num_timesteps, num_proposals, track_candidate, inference_time_range
        shape = (batch, self.num_proposals, 4)
        cur_bboxes = torch.randn(shape, device=self.device, dtype=self.dtype) * self.sigma_max
        ref_t_list = []
        track_t_list = []
        #print(self.sigmas[40])
        total_time = 0
        if ref_targets is None or self.track_candidate == 0:
            #print(1)
            ref_bboxes = torch.randn(shape, device=self.device) * self.sigma_max
            for i in range(batch):
                t = torch.randint(self.n_steps - self.inference_time_range, self.n_steps, (2,),
                                  device=self.device).long()
                if dynamic_time:
                    ref_t, track_t = self.n_steps - t[0] - 1, self.n_steps - t[1] - 1
                else:
                    ref_t, track_t = self.n_steps - t[0] - 1, self.n_steps - t[0] - 1
                ref_t_list.append(ref_t)
                track_t_list.append(track_t)
        else:
            # 每一个批次里有多少框，存入nlabel
            #print(2)
            labels = ref_targets[..., :5]
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
            shape = (batch, self.num_proposals, 4)
            diffused_boxes = []
            cur_diffused_boxes = []
            for batch_idx, num_gt in enumerate(nlabel):
                gt_bboxes_per_image = box_cxcywh_to_xyxy(labels[batch_idx, :num_gt])
                # images_whwh=torch.tensor([w, h, w, h], dtype=self.dtype, device=self.device)[None,:].expand(2*b,4)
                # 确定真值框的位置
                image_size_xyxy = images_whwh[batch_idx]
                gt_boxes = gt_bboxes_per_image / image_size_xyxy
                # cxcywh
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

                if batch_idx == 0:
                    ref_t = self.n_steps - consistency_t - 1
                    #ref_t = ref_t - 1
                    track_t = self.n_steps - consistency_t - 1
                    #track_t = track_t - 1
                else:
                    ref_t = self.n_steps - consistency_t - 1
                    #ref_t = ref_t - 1
                    track_t = self.n_steps - consistency_t - 1
                    #track_t = track_t - 1
                    self.track_candidate = 4
                # return diff_boxes, noise, select_mask
                #print(self.sigmas)
                #print(ref_t)
                #print(self.sigmas[ref_t])
                #print(track_t)
                #print(self.sigmas[track_t])
                x_start, d_boxes, d_noise, ref_label, d_tn1, d_tn, time_cond_n1, time_cond_n = self.prepare_consistency_concat(gt_boxes, ref_t)
                diffused_boxes.append(d_boxes)
                ref_t_list.append(ref_t)
                x_start, d_boxes, d_noise, ref_label, d_tn1, d_tn, time_cond_n1, time_cond_n = self.prepare_consistency_concat(gt_boxes, track_t, ref_label)
                cur_diffused_boxes.append(d_boxes)
                track_t_list.append(track_t)
            ref_bboxes = torch.stack(diffused_boxes)
            cur_bboxes = torch.stack(cur_diffused_boxes)
            self.sampling_timesteps=1 #1
        sampling_timesteps, eta = self.sampling_timesteps, self.ddim_sampling_eta
        #print(sampling_timesteps)
        #print(t)
        # 此处sampling_timestep已经设定好了为1000
        def get_time_pairs(t, sampling_timesteps):
            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            #print(t)
            times = torch.linspace(t, self.n_steps, steps=sampling_timesteps + 1)
            #print(times)
            times = list(reversed(times.int().tolist()))
            #print(times)
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            return time_pairs

        ref_t_time_pairs_list = torch.tensor([get_time_pairs(t, sampling_timesteps) for t in ref_t_list],
                                             device=self.device, dtype=torch.long)
        track_t_time_pairs_list = torch.tensor([get_time_pairs(t, sampling_timesteps) for t in track_t_list],
                                               device=self.device, dtype=torch.long)

        bboxes = torch.cat([ref_bboxes, cur_bboxes], dim=0)

        #print('initial ref_bbox: ', ref_bboxes.shape)
        #print('initial ref_bbox: ', cur_bboxes.shape)
        # (batch,sampling_timesteps,2)
        #print('initial bbox: ', bboxes.shape)


        B, N, _ = bboxes.shape
        s_in = bboxes.new_ones(B)

        x_start = None
        x_boxes = bboxes

        #print("ref_t_time_pairs_list:",ref_t_time_pairs_list)
        #print("track_t_time_pairs_list", track_t_time_pairs_list)
        # for (ref_time, ref_time_next),(cur_time, cur_time_next) in zip(ref_time_pairs,cur_time_pairs):
        for sampling_timestep in range(sampling_timesteps):
            #print(ref_t_time_pairs_list[:,sampling_timestep,0])
            """
            if(sampling_timestep == sampling_timesteps - 1):
                sigma_1 = self.sigmas[self.n_steps]
                sigma_2 = self.sigmas[self.n_steps]
                ref_time_cond = self.n_steps
                cur_time_cond = self.n_steps
            else:
            """
            ref = ref_t_time_pairs_list[:, sampling_timesteps - sampling_timestep - 1, 1]
            track = track_t_time_pairs_list[:, sampling_timesteps - sampling_timestep - 1, 1]
            ref = ref.to(self.device)
            track = track.to(self.device)
            self.sigmas = self.sigmas.to(self.device)
            sigma_1 = self.sigmas[ref]

            sigma_2 = self.sigmas[track]
            ref_time_cond = ref_t_time_pairs_list[:, sampling_timesteps - sampling_timestep - 1, 1]
            #print("ref_time_cond",ref_time_cond)
            #print("sigma_1", sigma_1)
            cur_time_cond = track_t_time_pairs_list[:, sampling_timesteps - sampling_timestep - 1, 1]
            #print("cur_time_cond", cur_time_cond)
                #print("sigma_2", sigma_2)

            sigma = torch.cat([sigma_1, sigma_2], dim=0)
            sigma = sigma.to(self.device)
            time = s_in * sigma
            #print("time:",time)
            is_last = sampling_timestep == (sampling_timesteps - 1)


            time_cond = torch.cat([ref_time_cond, cur_time_cond], dim=0)
            #print("time_cond:",time_cond)

            self_cond = x_start if self.self_condition else None
            #backbone_feats, images_whwh, boxes_test, time, time_cond
            x_boxes, x_start, outputs_class, outputs_coord, outputs_score, association_time = self.model_consistency_predictions(backbone_feats,
                                                                images_whwh, x_boxes, time, time_cond,fix_bboxes=False)
            #print('x_boxes: ', x_boxes.shape)
            #print('x_start: ', x_start.shape)
            #print('outpus_class: ', outputs_class.shape)
            #print('outpus_coord: ', outputs_coord.shape)

            total_time += association_time
            #pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            #print(1)

            if is_last:
                #print(1)
                x_boxes = x_start
                continue

            if self.box_renewal:  # filter
                # print('starting renewal...')
                #print(1)
                remain_list = []
                pre_remain_bboxes = []
                pre_remain_x_start = []
                pre_x_boxes = []
                cur_x_boxes = []
                cur_remain_x_start = []
                cur_remain_pred_noise = []
                for i in range(batch):
                    threshold = 0.6  ####0.6
                    score_per_image = outputs_score[-1][i]  # [1000, 1]
                    value, _ = torch.max(score_per_image, -1, keepdim=False)  # [1000]
                    keep_idx = value >= threshold
                    num_remain = torch.sum(keep_idx)
                    remain_list.append(num_remain)
                    pre_x_boxes.append(x_boxes[i, keep_idx, :])
                    cur_x_boxes.append(x_boxes[i + batch, keep_idx, :])
                    pre_remain_x_start.append(x_start[i, keep_idx, :])
                    cur_remain_x_start.append(x_start[i + batch, keep_idx, :])

                #print('renewal ended:')
                #print('pre_x_boxes len:', len(pre_x_boxes))
                #print('pre_x_boxes 0 1:', pre_x_boxes[0].shape, pre_x_boxes[1].shape)
                #print('cur_x_boxes 0 1:', cur_x_boxes[0].shape, cur_x_boxes[1].shape)
                x_start = pre_remain_x_start + cur_remain_x_start  # list[4] = list[2] + list[2]
                x_boxes = pre_x_boxes + cur_x_boxes

                #pred_noise = pre_remain_pred_noise + cur_remain_pred_noise
                #print("len x_boxesnow:", len(x_boxes))
                #print('x_boxes 0 1 2 3:', x_boxes[0].shape, x_boxes[1].shape,
                      #x_boxes[2].shape, x_boxes[3].shape)

            def Consistency(sampling_timestep, bboxes, x_start, sigma):
                #print(sampling_timestep)
                times = sampling_timestep[:,0]

                #print("times:",times)
                sigma_t = torch.tensor([self.sigmas[time] for time in times], dtype=self.dtype,
                                     device=self.device)
                #print("sigma_t",sigma_t)
                #print('\n')
                #print("sigma",sigma)
                #print("bboxes:1",bboxes[0])
                if self.box_renewal:
                    #print('now in Consystency...')
                    #print("before bboxes:", bboxes[0].shape, bboxes[1].shape)
                    for i in range(batch):
                        #print(bboxes[i])
                        #print(sigma[i])
                        d = (bboxes[i] - x_start[i]) / append_dims(sigma[i], bboxes[i].ndim).to(self.device)
                        #print(d)
                        dt = sigma_t[i] - sigma[i]
                        #print(dt)
                        # x_start_xyxys
                        # x_start_xyxys = (x_start_xyxy01 * 2. - 1.) * self.scale
                        # boxes_test_xyxy
                        bboxes[i] = bboxes[i] + d * dt

                        # 经过renewal后不足num_proposal, 此处通过randn补充
                        bboxes[i] = torch.cat(
                            (bboxes[i], torch.randn(self.num_proposals - remain_list[i], 4, device=self.device) * sigma_t[i]), dim=0)
                    #print("end bboxes:", bboxes[0].shape, bboxes[1].shape)
                    #print("bboxes", bboxes)

                else:
                    #noise = torch.randn_like(bboxes)
                    d = (bboxes - x_start) / append_dims(sigma, bboxes.ndim).to(self.device)
                    dt = self.sigmas[times] - sigma
                    # x_start_xyxys
                    # x_start_xyxys = (x_start_xyxy01 * 2. - 1.) * self.scale
                    # boxes_test_xyxy
                    bboxes = bboxes + d * dt

                return bboxes
            #print(ref_t_time_pairs_list)
            x_boxes[:batch] = Consistency(ref_t_time_pairs_list[:, sampling_timesteps - sampling_timestep - 1], x_boxes[:batch], x_start[:batch], sigma_1)
            x_boxes[batch:] = Consistency(track_t_time_pairs_list[:, sampling_timesteps - sampling_timestep - 1], x_boxes[batch:], x_start[batch:], sigma_2)

            if self.box_renewal:
                x_boxes = torch.stack(x_boxes)  # stack along axis=0  list->tensor

        box_cls = outputs_class[-1]
        box_pred = outputs_coord[-1]
        conf_score = outputs_score[-1]

        """
            return torch.cat([box_pred.view(2*batch,-1,4), box_cls.view(2*batch,-1,1)], dim=-1), 
            conf_score.view(batch,-1,1), total_time：这是返回语句，它将最终的输出组合为一个元组。具体来说：
            box_pred.view(2*batch, -1, 4) 会将 box_pred 重新排列成一个大小为 (2*batch, -1, 4) 的张量，其中 -1 表示自动计算的维度。
            box_cls.view(2*batch, -1, 1) 会将 box_cls 重新排列成一个大小为 (2*batch, -1, 1) 的张量，其中 -1 表示自动计算的维度。
            torch.cat([box_pred.view(2*batch, -1, 4), box_cls.view(2*batch, -1, 1)], dim=-1) 
            将上述两个张量在最后一个维度上进行连接，得到一个包含目标框坐标和类别概率的张量。
            conf_score.view(batch, -1, 1) 会将 conf_score 重新排列成一个大小为 (batch, -1, 1) 的张量，其中 -1 表示自动计算的维度。
            total_time 是一个记录总计算时间的值，它可能用于性能分析。
        """
        #print("box_pred:",box_pred)
        #print("box_pred.view(2 * batch, -1, 4):",box_pred.view(2 * batch, -1, 4))
        return torch.cat([box_pred.view(2 * batch, -1, 4), box_cls.view(2 * batch, -1, 1)], dim=-1), conf_score.view(
        batch, -1, 1), total_time


    # consistency model add noise
    def q_sample(self, x_start, t_n1, noise=None):
        # q = 0.5
        if noise is None:
            noise = torch.randn_like(x_start)
        return x_start + noise * t_n1


    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data ** 2 / (
                (sigma - self.sigma_min) ** 2 + self.sigma_data ** 2
        )
        c_out = (
                (sigma - self.sigma_min)
                * self.sigma_data
                / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        )
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out * 2, c_in / 2

    def consistency_function(self, model, features, boxes, t, time_cond, images_whwh, lost_features=None, fix_bboxes=None, training=False):

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, boxes.ndim) for x in self.get_scalings(t)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, boxes.ndim)
                for x in self.get_scalings_for_boundary_condition(t)
            ]

        c_in = c_in.to(self.device)
        #print(c_in)
        boxes = boxes.to(self.device)

        bboxes = c_in * boxes
        #print("boxes1:",bboxes)
        bboxes = torch.clamp(bboxes, min=-1, max=1)

        bboxes = (bboxes + 1) / 2.
        #print(1)
        bboxes = box_cxcywh_to_xyxy(bboxes)

        bboxes = bboxes * images_whwh[:, None, :]

        bs = len(bboxes) // 2
        #print("bs",torch.split(bboxes, bs, dim=0))
        #print(bs)
        #outputs_class, outputs_coord, outputs_score = self.head(backbone_feats, torch.split(bboxes, bs, dim=0), t, lost_features, fix_bboxes)
        # 此处应该接受当前采样时间步
        outputs_class, outputs_coord, outputs_score = model(features, torch.split(bboxes, bs, dim=0), time_cond, lost_features, fix_bboxes)

        outputs_coord = outputs_coord / images_whwh[:, None, :]

        denoised_boxes = torch.clamp(outputs_coord, min=0, max=1)

        denoised_boxes = denoised_boxes * images_whwh[:, None, :]

        return outputs_class, denoised_boxes, outputs_score

    def forward(self, features, mate_info, targets=None):

        mate_shape, mate_device, mate_dtype = mate_info
        self.device = mate_device
        self.dtype = mate_dtype
        # x format (pre_imgs,cur_imgs) (B,C,H,W)
        b, _, h, w = mate_shape

        images_whwh = torch.tensor([w, h, w, h], dtype=self.dtype, device=self.device)[None, :].expand(2 * b, 4)

        if not self.training:
            results = self.new_ddim_sample(features, images_whwh, targets, dynamic_time=False)  # new_ddim_sample
            return results
        # 到哪里了
        if self.training:
            # new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)
            #targets, x_boxes, noises, t = self.prepare_targets(targets, images_whwh)
            tar, targets, boxes_n1, noises, t_n1, t_n, time_cond_n1, time_cond_n = self.prepare_targets(targets, images_whwh)
            t_n1 = t_n1.squeeze(-1)
            t_n = t_n.squeeze(-1)
            time_cond_n1 = time_cond_n1.squeeze(-1)
            time_cond_n = time_cond_n.squeeze(-1)
            #print(self.num_proposals)
            def euler_solver(samples, t, next_t, x0):
                x = samples
                denoiser = x0
                dims = x0.ndim
                d = (x - denoiser) / append_dims(t, dims)
                # x = (x * 2. - 1.) * self.scale
                # x_t = x_start + noise * append_dims(t, dims)
                samples = x + d * append_dims(next_t - t, dims)
                return samples
            # t[b:]=t[:b]
            boxes_n = euler_solver(boxes_n1, t_n1, t_n, tar)
            #pre_x_boxes_n1, cur_x_boxes_n1 = torch.split(boxes_n1, b, dim=0)
            # 从中间隔开
            #pre_x_boxes, cur_x_boxes = torch.split(boxes_n, b, dim=0)

            outputs_class, outputs_coord, outputs_score = self.consistency_function(self.head, features, boxes_n1, t_n1,
                                                                     time_cond_n1, images_whwh, training=self.training)
            # outputs_coord = torch.clamp(outputs_coord, min=0, max=255)
            outputs_class_minus, outputs_coord_minus, outputs_score_minus = self.consistency_function(self.head, features, boxes_n, t_n,
                                                                                 time_cond_n, images_whwh, training=self.training)

            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                      'pred_scores': outputs_score[-1]}

            output_minus = {'pred_logits': outputs_class_minus[-1], 'pred_boxes': outputs_coord_minus[-1],
                      'pred_scores': outputs_score_minus[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_scores': c}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_score[:-1])]
                output_minus['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_scores': c}
                                         for a, b, c in zip(outputs_class_minus[:-1], outputs_coord_minus[:-1], outputs_score_minus[:-1])]
            loss_dict1 = self.criterion(output, targets)  # self.criterion(output, output_minus, targets)
            loss_dict2 = self.criterion(output_minus, targets)  # self.criterion(output, output_minus, targets)
            # add consistency loss for pred_boxes and pred_boxes_minus
            for k in loss_dict1.keys():
                loss_dict1[k] = loss_dict1[k] + loss_dict2[k]

            weight_dict = self.criterion.weight_dict
            # weight_dict["loss_consistency"] = snrs

            for k in loss_dict1.keys():
                if k in weight_dict:
                    loss_dict1[k] *= weight_dict[k]
            return loss_dict1

    def prepare_consistency_concat(self, gt_boxes, t, ref_mask=None):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        if self.training:
            self.track_candidate = 1
        t = torch.full((1,), t, device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device, dtype=self.dtype)
        select_mask = None
        num_gt = gt_boxes.shape[0] * self.track_candidate
        # 如果存在目标框（num_gt 大于零）：使用 torch.repeat_interleave 函数对 gt_boxes 进行重复。
        # 每个现有的目标框将被重复 self.track_candidate 次。这可以用于生成多个重复的目标框，以便进行跟踪等操作。
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=self.dtype, device=self.device)
            num_gt = 1
        else:
            gt_boxes = torch.repeat_interleave(gt_boxes, torch.tensor([self.track_candidate] * gt_boxes.shape[0],
                                                                      device=self.device), dim=0)

        if num_gt < self.num_proposals:
            # 创建一个形状为 (self.num_proposals - num_gt, 4) 的张量，其中的值是从标准正态分布中随机生成的。
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device,
                                          dtype=self.dtype) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            # box_placeholder=torch.clip(torch.poisson(torch.clip(box_placeholder*5,min=0)),min=1,max=10)/10
            # box_placeholder=torch.nn.init.uniform_(box_placeholder, a=0, b=1)
            # box_placeholder=torch.ones_like(box_placeholder)
            # box_placeholder[:,:2]=box_placeholder[:,:2]/2
            box_placeholder[:, 2:4] = torch.clip(box_placeholder[:, 2:4], min=1e-4)
            # 这行代码将真实目标框 gt_boxes 与虚拟目标框 box_placeholder 连接起来，生成一个新的张量
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            if ref_mask is not None:
                select_mask = ref_mask
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes


        num_scales = self.n_steps

        # t_n+1
        t_n1 = self.sigma_max ** (1 / self.rho) + t / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t_n1 = (t_n1 ** self.rho).to(self.device)

        # t_n
        t_n = self.sigma_max ** (1 / self.rho) + (t + 1) / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t_n = (t_n ** self.rho).to(self.device)

        x_start = (x_start * 2. - 1.)

        if self.noise_strategy == "xy":
            noise[:, 2:] = 0
        # noise sample
        x_t_n1 = self.q_sample(x_start=x_start, t_n1=t_n1, noise=noise)
        """
        if self.training:
            # x=x_start

            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.

            diff_boxes = box_cxcywh_to_xyxy(x)
        else:
            diff_boxes = x
        """
        return x_start, x_t_n1, noise, select_mask, t_n1, t_n, t, t + 1

    def prepare_targets(self, targets, images_whwh):
        labels = targets[..., :5]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        new_targets = []
        diffused_boxes_n = []
        noises = []
        ts1 = []
        ts = []
        ts_n1 = []
        ts_n = []
        tar = []
        select_mask = {}
        # select_t={}
        # select_gt_boxes={}
        for batch_idx, num_gt in enumerate(nlabel):
            target = {}
            gt_bboxes_per_image = box_cxcywh_to_xyxy(labels[batch_idx, :num_gt, 1:5])
            gt_classes = labels[batch_idx, :num_gt, 0]
            image_size_xyxy = images_whwh[batch_idx]
            gt_boxes = gt_bboxes_per_image / image_size_xyxy
            # cxcywh
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            x_gt_boxes = gt_boxes
            d_t = torch.randint(0, self.n_steps, (1,), device=self.device).long()[0]

            x_start, d_boxes_n, d_noise, d_mask, d_tn1, d_tn, time_cond_n1, time_cond_n = self.prepare_consistency_concat(x_gt_boxes, d_t,
                                                                     select_mask.get(batch_idx % (len(nlabel) // 2),None))

            # per_img的mask需要存储给下一帧使用
            if d_mask is not None:
                select_mask[batch_idx % (len(nlabel) // 2)] = d_mask
            # if d_t is not None:
            #     select_t[batch_idx%(len(nlabel)//2)]=d_t
            # if select_gt_boxes.get(batch_idx%(len(nlabel)//2),None) is None:
            #     select_gt_boxes[batch_idx%(len(nlabel)//2)]=gt_boxes
            tar.append(x_start)
            diffused_boxes_n.append(d_boxes_n)
            noises.append(d_noise)
            ts1.append(d_tn1)
            ts.append(d_tn)
            ts_n1.append(time_cond_n1)
            ts_n.append(time_cond_n)

            target["labels"] = gt_classes.long()
            target["boxes"] = gt_boxes
            target["boxes_xyxy"] = gt_bboxes_per_image
            target["image_size_xyxy"] = image_size_xyxy
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt
            new_targets.append(target)

        # XYXY01，XYXY255，XYXY01，XYXY01，01，01
        return torch.stack(tar), new_targets, torch.stack(diffused_boxes_n), torch.stack(
            noises), torch.stack(ts1), torch.stack(ts), torch.stack(ts_n1), torch.stack(ts_n)



