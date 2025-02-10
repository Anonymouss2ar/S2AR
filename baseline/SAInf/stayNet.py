import copy
import logging
import time

import math
import numpy as np
import torch
import torch.nn as nn


from stay_area.util.cal_distance import distance



class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, x):
        attn_output = self.multi_head_attention(x, x, x)[0]
        x = self.norm1(x + attn_output)
        feed_forward_output = self.feed_forward(x)
        return self.norm2(x + feed_forward_output)


class cross_attention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(cross_attention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, st_emb, grid_emb):
        attn_output = self.multi_head_attention(st_emb, grid_emb, grid_emb)[0]
        x = self.norm1(st_emb + attn_output)
        feed_forward_output = self.feed_forward(x)
        return self.norm2(x + feed_forward_output)


class StayNet(nn.Module):
    def __init__(self, input_dim, num_heads, num_blocks, dropout=0.1):
        super(StayNet, self).__init__()
        self.transformer = nn.Sequential(*[TransformerBlock(input_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Linear(input_dim, 1)
        self.grid_linear = nn.Linear(14, input_dim)
        self.grid_gps_linear = nn.Linear(4, input_dim // 4)
        self.traj_gps_linear = nn.Linear(4, input_dim // 4)
        self.node_linear = nn.Linear(128, input_dim // 4)
        self.time_emb = nn.Embedding(48, input_dim // 4)
        self.time_of_weekend_linear = nn.Linear(1, input_dim // 4)
        self.weather_emb = nn.Embedding(12, input_dim // 4)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.inputdim = input_dim
        self.multi_head_attention = cross_attention(input_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, candate_grid, grid_gps, traj_gps, node_emb, traj_time, iweekend, weather_data):
        grid_emb = self.grid_linear(candate_grid)
        grid_emb_atten = self.transformer(grid_emb)
        spatial_emb = self.grid_gps_linear(grid_gps) + self.traj_gps_linear(traj_gps) + self.node_linear(node_emb)

        org_bs, org_l, org_embedding_dim = traj_time.shape
        traj_time_reshape = traj_time.reshape(-1, org_embedding_dim)
        time_emb = self.time_emb(traj_time_reshape)
        embedded_tensor_time = time_emb.reshape(org_bs, org_l, self.inputdim // 4)
        isweekend = self.time_of_weekend_linear(iweekend)
        org_bs, org_l, org_embedding_dim = weather_data.shape
        weather_data_reshape = weather_data.reshape(-1, org_embedding_dim)
        weather_emb = self.weather_emb(weather_data_reshape)
        embedded_tensor_weather = weather_emb.reshape(org_bs, org_l, self.inputdim // 4)
        st_emb = torch.cat([spatial_emb, embedded_tensor_time, isweekend, embedded_tensor_weather], dim=-1)
        st_emb_ffn = self.feed_forward1(st_emb)
        cand_multi_output = self.multi_head_attention(st_emb_ffn, grid_emb_atten)
        final_out_put = torch.cdist(cand_multi_output, grid_emb_atten, p=2)
        final_output = self.softmax(final_out_put)
        return final_output

    def test_model(self, candate_grid, grid_gps, traj_gps, node_emb, traj_time, iweekend, weather_data):
        grid_emb = self.grid_linear(candate_grid)
        grid_emb_atten = self.transformer(grid_emb)
        spatial_emb = self.grid_gps_linear(grid_gps) + self.traj_gps_linear(traj_gps) + self.node_linear(node_emb)

        org_bs, org_l, org_embedding_dim = traj_time.shape
        traj_time_reshape = traj_time.reshape(-1, org_embedding_dim)
        time_emb = self.time_emb(traj_time_reshape)
        embedded_tensor_time = time_emb.reshape(org_bs, org_l, self.inputdim // 4)
        isweekend = self.time_of_weekend_linear(iweekend)
        org_bs, org_l, org_embedding_dim = weather_data.shape
        weather_data_reshape = weather_data.reshape(-1, org_embedding_dim)
        weather_emb = self.weather_emb(weather_data_reshape)
        embedded_tensor_weather = weather_emb.reshape(org_bs, org_l, self.inputdim // 4)
        st_emb = torch.cat([spatial_emb, embedded_tensor_time, isweekend, embedded_tensor_weather], dim=-1)
        st_emb_ffn = self.feed_forward1(st_emb)
        cand_multi_output = self.multi_head_attention(st_emb_ffn, grid_emb_atten)
        cand_multi_output = torch.sum(cand_multi_output, dim=1) / cand_multi_output.shape[1]
        grid_emb_atten = torch.sum(grid_emb_atten, dim=1) / grid_emb_atten.shape[1]
        final_output_cross_sim=torch.cdist(cand_multi_output,grid_emb_atten,p=2)
        final_output = self.softmax(final_output_cross_sim)
        return final_output
class NoamOpt:
    "Optim wrapper that implements rate."
    # 学习率动态变化的adam优化
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    def rate(self, step=None):
        """Implement `lrate` above
        改变学习率
        """
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))
class loss_cal:
    def __init__(self, opt):
        self.opt = opt
        self.cl = nn.CrossEntropyLoss()
        self.bueloss = nn.BCEWithLogitsLoss()
    def cal(self, outdata, true_data, total_length):
        total_loss = []
        for i in range(outdata.shape[0]):  #取每行
            cur_pre_data = outdata[i]
            cur_length = total_length[i]
            cur_pre_data = cur_pre_data[0,:cur_length]
            cur_truth_data = true_data[i,: , :cur_length][0]
            bceloss = self.bueloss(cur_pre_data, cur_truth_data)
            total_loss.append(bceloss)
        loss = sum(total_loss) / len(total_loss)
        loss.backward()
        if self.opt is not None:
            self.opt.step()  # 更新梯度
            self.opt.optimizer.zero_grad()  # 梯度清零
        return loss.item()


def train_epoch(train_data_loader, model, loss_cal, device):
    model.train()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    loss_list = []
    logging.info("total tokens: {}".format(len(train_data_loader)))
    for l, batch in enumerate(train_data_loader):
        camera_traj_gps, camera_traj_time, \
            camera_grid_gps, \
            new_truth_data, \
            weather_data, node_emb, new_candidate_region, isweekend, total_cand_length = batch
        camera_traj_gps = camera_traj_gps.to(device)
        camera_traj_time = camera_traj_time.to(device)
        camera_grid_gps = camera_grid_gps.to(device)
        weather_data = weather_data.to(device)
        node_emb = node_emb.to(device)
        new_candidate_region = new_candidate_region.to(device)
        isweekend = isweekend.to(device)
        output = model(new_candidate_region, camera_grid_gps, camera_traj_gps, node_emb, camera_traj_time, isweekend,
                       weather_data)
        output = output.cpu()
        loss = loss_cal.cal(output, new_truth_data, total_cand_length)
        loss_list.append(loss)
        if l % 50 == 0:
            logging.info("Epoch Step: %d Loss: %f mean loss: %f" %
                         (l, loss, sum(loss_list) / len(loss_list)))
    return loss_list

def test_epoch(test_data_loader, model, device,node2gps):
    model.eval()
    logging.info("total tokens: {}".format(len(test_data_loader)))
    out_put_data = []
    total_truth_data = []
    final_pre_id_list = []
    total_cand_len_list = []
    with torch.no_grad():
        for l, batch in enumerate(test_data_loader):
            camera_traj_gps, camera_traj_time, \
                camera_grid_gps, \
                weather_data, node_emb, new_candidate_region, isweekend, total_cand_length, pre_id_list, \
                truth_data = batch
            camera_traj_gps = camera_traj_gps.to(device)
            camera_traj_time = camera_traj_time.to(device)
            camera_grid_gps = camera_grid_gps.to(device)
            weather_data = weather_data.to(device)
            node_emb = node_emb.to(device)
            new_candidate_region = new_candidate_region.to(device)
            isweekend = isweekend.to(device)
            pre_data = model.test_model(new_candidate_region, camera_grid_gps, camera_traj_gps, node_emb,
                                        camera_traj_time, isweekend, weather_data)
            final_out_put_data = pre_data.cpu()
            out_put_data.extend(np.array(final_out_put_data.reshape(final_out_put_data.shape[0],-1)))
            total_truth_data.extend(truth_data)
            final_pre_id_list.extend(pre_id_list)
            total_cand_len_list.extend(total_cand_length)
    final_output = np.array(out_put_data)
    test_model_method(total_truth_data, final_pre_id_list, final_output, total_cand_len_list,node2gps)


def get_top_key(matrix, K, axis):

    total_len = len(matrix)
    total_top_key_idx = []
    for i in range(total_len):
        cur_data = matrix[i][:K[i]]
        sorted_cur_data = np.argsort(-cur_data)
        topk_idx = sorted_cur_data
        total_top_key_idx.append(topk_idx)
    return total_top_key_idx


def test_model_method(total_truth_data, pre_id_list, out_put_data, total_length, node2gps):
    total_len = len(out_put_data)
    top_k_pred_matrix = get_top_key(out_put_data, K=total_length, axis=1)
    top_k_pred_matrix = np.array(top_k_pred_matrix)
    mr = []
    rr = []
    mrr = []
    rmse = []
    aed = []
    for id_index, each_traj_id in enumerate(pre_id_list):
        top_k_truth = total_truth_data[id_index]
        top_k_pred_id_list = pre_id_list[id_index]
        top_key_model_output = top_k_pred_matrix[id_index]
        top_k_pred_truth = []
        for pred_index in top_key_model_output:
            if pred_index < len(top_k_pred_id_list) - 1:
                top_k_pred_truth.append(top_k_pred_id_list[pred_index])
            else:
                top_k_pred_truth.append(top_k_pred_id_list[-1])
        if top_k_truth in top_k_pred_truth:
            truth_idx = top_k_pred_truth.index(top_k_truth)
        else:
            truth_idx = len(top_k_pred_truth) - 1
        mr.append(truth_idx)
        rr.append(truth_idx / len(top_k_pred_id_list))
        mrr.append(1 / (truth_idx + 1))
        mse = (truth_idx - 1) ** 2
        rmse.append(mse)
        truth_gps = node2gps[str(top_k_truth)]
        pre_truth_gps = node2gps[str(top_k_pred_truth[0])]
        abs_distance = abs(distance(truth_gps[1], truth_gps[0], pre_truth_gps[1], pre_truth_gps[0]))
        aed.append(abs_distance)

    logging.info("total mr: {}".format(sum(mr) / len(mr)))  #越低越好
    logging.info("total rr: {}".format(sum(rr) / len(rr)))  #越低越好
    logging.info("total mrr: {}".format(sum(mrr) / len(mrr)))  #越接近1效果越好
    logging.info("total rmse: {}".format(math.sqrt(sum(rmse) / len(rmse))))  #越小越好
    logging.info("total aed: {}".format(sum(aed) / len(aed)))
