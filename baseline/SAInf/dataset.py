import json
import pickle
from datetime import datetime

import math
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from stay_area.util.cal_distance import distance

class StayAreaDataset(Dataset):
    def __init__(self, final_traj_data):
        self.camera_traj_gps =final_traj_data['camera_traj_gps']
        self.camera_traj_time=final_traj_data['camera_traj_time']
        self.camera_traj_node = final_traj_data['camera_traj_node']
        self.camera_grid_gps=final_traj_data['camera_grid_gps']
        self.candat_region=final_traj_data['candat_region']
        self.truth_data=final_traj_data['truth_data']
        self.weather_data=final_traj_data['weather_data']
        self.node_emb=final_traj_data['node_emb']
        self.candat_ragion_grid=final_traj_data['candat_ragion_grid']
        self.isweekend=final_traj_data['isweekend']

    def __len__(self):
        return len(self.camera_traj_gps)

    def __getitem__(self, idx):
        cmaera_traj_gps=self.camera_traj_gps[idx]
        camera_traj_time=self.camera_traj_time[idx]
        camera_traj_node=self.camera_traj_node[idx]
        camera_grid_gps=self.camera_grid_gps[idx]
        candat_region=self.candat_region[idx]
        truth_data=self.truth_data[idx]
        weather_data=self.weather_data[idx]
        node_emb=self.node_emb[idx]
        candidate_region_grid=self.candat_ragion_grid[idx]
        isweekend=self.isweekend[idx]
        return (cmaera_traj_gps, camera_traj_time,
                camera_traj_node, camera_grid_gps,
                candat_region, truth_data, weather_data,
                node_emb, candidate_region_grid, isweekend)
def get_data(input_data,grid,node_map,weather,node_emb):
    total_traj_keys=input_data.keys()
    final_truth_data=[]
    final_camera_traj_gps=[]
    final_candat_region=[]
    final_camera_traj_time=[]
    final_isweekend=[]
    final_weather_data=[]
    final_camera_traj_node=[]
    final_node_emb=[]
    final_camera_grid_gps=[]
    final_candat_ragion_grid=[]
    for each_key in total_traj_keys:
        cur_data=input_data[each_key]
        cur_stay_area=cur_data['stay_area']
        cur_candat_region=cur_data['candate_ragion']
        cur_timestamp=cur_data['timestamp']
        cur_gps=cur_data['gps']
        cur_node=cur_data['node']
        for j in range(len(cur_stay_area)):
            tmp_stay_truth=cur_stay_area[j]
            tmp_candat_region=cur_candat_region[tmp_stay_truth[0]]
            if len(tmp_candat_region)>3:
                tmp_gps=cur_gps[tmp_stay_truth[0]]
                tmp_timestamp=cur_timestamp[tmp_stay_truth[0]]
                new_tmp_candate_region = []
                new_pre_id_list=[]
                for each_candata_ragion in tmp_candat_region:
                    hang, lie = each_candata_ragion
                    cur_grid_data = grid[hang][lie]
                    new_tmp_candate_region.append(cur_grid_data)
                    new_pre_id_list.append(cur_grid_data[0])
                if tmp_stay_truth[1] in new_pre_id_list:
                    final_candat_ragion_grid.append(new_tmp_candate_region)
                    final_truth_data.append(tmp_stay_truth[1])
                    final_camera_traj_gps.append([tmp_gps[0][0],tmp_gps[0][1],tmp_gps[1][0],tmp_gps[1][1]])
                    final_candat_region.append(tmp_candat_region)
                    start_time=tmp_timestamp[0]
                    all_date=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time))
                    today_date=all_date.split(" ")[0]
                    time_sloat=all_date.split(" ")[1]
                    start_hour=time_sloat.split(":")[0]
                    end_time = tmp_timestamp[0]
                    end_time_all_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
                    # today_date = all_date.split(" ")[0]
                    end_time_sloat = end_time_all_date.split(" ")[1]
                    end_hour = end_time_sloat.split(":")[0]
                    final_camera_traj_time.append([int(start_hour)+int(end_hour)])
                    cur_weather=weather[today_date]
                    dt = datetime.fromtimestamp(start_time)
                    if dt.weekday() >= 5:
                        final_isweekend.append([1])
                    else:
                        final_isweekend.append([0])
                    final_weather_data.append([cur_weather])
                    tmp_node=cur_node[tmp_stay_truth[0]]
                    start_node=node_map[tmp_node[0]]
                    end_node=node_map[tmp_node[1]]
                    final_camera_traj_node.append([start_node+end_node])
                    start_node_emb=node_emb[int(start_node)]
                    end_node_emb=node_emb[int(end_node)]
                    final_node_emb.append([start_node_emb+end_node_emb])
                    start_grid_id,start_grid_gps=get_grid_data(grid,tmp_gps[0][0],tmp_gps[0][1])
                    end_grid_id, end_grid_gps = get_grid_data(grid, tmp_gps[1][0], tmp_gps[1][1])
                    final_camera_grid_gps.append([start_grid_gps[0],start_grid_gps[1],end_grid_gps[0],end_grid_gps[1]])
    final_traj_data={}
    final_traj_data['truth_data']=final_truth_data
    final_traj_data['camera_traj_gps']=final_camera_traj_gps
    final_traj_data['candat_region']=final_candat_region
    final_traj_data['camera_traj_time']=final_camera_traj_time
    final_traj_data['isweekend']=final_isweekend
    final_traj_data['weather_data']=final_weather_data
    final_traj_data['camera_traj_node']=final_camera_traj_node
    final_traj_data['node_emb']=final_node_emb
    final_traj_data['camera_grid_gps']=final_camera_grid_gps
    final_traj_data['candat_ragion_grid']=final_candat_ragion_grid
    with open("./data/train/train_data.pkl", "wb") as f:
        pickle.dump(final_traj_data,f)
        f.close()
def get_grid_data(grid,lon,lat):
    x_min = 116.80
    y_min = 35.8
    x_size = 1000
    y_size = 1000
    x_length = distance(lat, x_min, lat, lon)
    y_length = distance(y_min, lon, lat, lon)
    x_id = int(x_length // x_size)
    y_id = int(y_length // y_size)
    cur_grid = grid[str(x_id)][str(y_id)]
    return cur_grid[0],cur_grid[1]
def cal_tfidf(input_data,camera_gps):
    total_len=len(input_data)
    new_total_grid_region=[]
    total_pre_id=[]
    for i in range(total_len):#遍历
        cur_grid=input_data[i]
        cur_gps=camera_gps[i]
        pre_id=[]
        total_cur_tf=[]
        start_lon=cur_gps[0]
        start_lat=cur_gps[1]
        end_lon=cur_gps[2]
        end_lat=cur_gps[3]
        distance_feature=[]
        for j in range(len(cur_grid)):
            cur_cand_ragion_tf=cur_grid[j][4]
            pre_id.append(cur_grid[j][0])
            grid_gps=cur_grid[j][1]
            aoi_total_key=cur_cand_ragion_tf.keys()
            cur_tf=[]
            for each_aoi_key in cur_cand_ragion_tf.keys():
                cur_tf.append(cur_cand_ragion_tf[each_aoi_key])
            sum_tf=sum(cur_tf)
            if sum_tf==0:
                for jjj in range(len(cur_tf)):
                    cur_tf[jjj]=0
            else:
                for jjj in range(len(cur_tf)):
                    cur_tf[jjj]=cur_tf[jjj]/sum_tf
            total_cur_tf.append(cur_tf)#l*13
            grid2camera_dist=distance(grid_gps[1],grid_gps[0],start_lat,start_lon)+distance(grid_gps[1],grid_gps[0],end_lat,end_lon)
            distance_feature.append(grid2camera_dist)
        total_aoi_each_list=[]
        total_pre_id.append(pre_id)
        for each_aoi_key in aoi_total_key:
            tmp_each_cand_aoi_list=[]
            for j in range(len(cur_grid)):
                cur_cand_ragion_tf=cur_grid[j][4]
                tmp_each_cand_aoi_list.append(cur_cand_ragion_tf[each_aoi_key])
            sum_idf_each_doc=sum(tmp_each_cand_aoi_list)
            for jjj in range(len(tmp_each_cand_aoi_list)):
                cur_idf_data=tmp_each_cand_aoi_list[jjj]
                tmp_each_cand_aoi_list[jjj]=math.log(len(cur_grid)/(sum_idf_each_doc-cur_idf_data+1))
            total_aoi_each_list.append(tmp_each_cand_aoi_list)#13*l
        new_tmp_grid_feature=[]
        for each_l in range(len(total_cur_tf)):
            new_each_cand_feature=[]
            for each_feature in range(len(total_cur_tf[each_l])):
                cur_tf_data=total_cur_tf[each_l][each_feature]
                cur_idf_data=total_aoi_each_list[each_feature][each_l]
                cur_tfidf=cur_tf_data*cur_idf_data
                new_each_cand_feature.append(cur_tfidf)
            cur_distance_feature = distance_feature[each_l]
            new_each_cand_feature.append(cur_distance_feature)
            new_tmp_grid_feature.append(new_each_cand_feature)
        new_total_grid_region.append(new_tmp_grid_feature)
    return new_total_grid_region,total_pre_id

def merge(sequences):#补零
    lengths = [len(seq) for seq in sequences]#计算sequence里每个的长度然后组成一个list
    # dim = sequences[0].size(1)  # get dim for each sequence
    padded_seqs = torch.zeros(len(sequences), max(lengths),14,dtype=torch.float32)#全变成0然后补值
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = torch.tensor(seq[:end],dtype=torch.float32)
    return padded_seqs, lengths
def merge_each(sequences,dim,total_cand_length):#补零
    lengths = [len(seq) for seq in sequences]#计算sequence里每个的长度然后组成一个list
    # lengths = total_cand_length
    # dim = sequences[0].size(1)  # get dim for each sequence
    padded_seqs = torch.zeros(len(sequences), 1,dim,dtype=torch.float32)#全变成0然后补值
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = torch.tensor(seq,dtype=torch.float32)
    return padded_seqs, lengths
def merge_true_data(sequences,dim):
    lengths = [len(seq) for seq in sequences]  # 计算sequence里每个的长度然后组成一个list
    # lengths = total_cand_length
    # dim = sequences[0].size(1)  # get dim for each sequence
    padded_seqs = torch.zeros(len(sequences), 1, max(lengths), dtype=torch.float32)  # 全变成0然后补值
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i,:, :end] = torch.tensor(seq, dtype=torch.float32)
    return padded_seqs, lengths
def merge_int(sequences,dim,total_cand_length):#补零
    lengths = [len(seq) for seq in sequences]#计算sequence里每个的长度然后组成一个list
    # dim = sequences[0].size(1)  # get dim for each sequence
    # lengths = total_cand_length
    padded_seqs = torch.zeros(len(sequences), 1,dim,dtype=torch.int64)#全变成0然后补值
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = torch.tensor(seq,dtype=torch.int64)
    return padded_seqs, lengths
def collate_fn(data):#得到的grid是b*l*15
    (cmaera_traj_gps, camera_traj_time,
                camera_traj_node, camera_grid_gps,
                candat_region, truth_data, weather_data,
                node_emb, candidate_region_grid, isweekend)=zip(*data)
    new_candidate_region,total_pre_id=cal_tfidf(candidate_region_grid,cmaera_traj_gps)
    # new_candidate_region=torch.tensor(new_candidate_region)
    new_candidate_region,total_cand_length=merge(new_candidate_region)
    for jj in range(len(total_pre_id)):
        cur_pre_id=total_pre_id[jj]
        cur_truth_data=truth_data[jj]
        for ii in range(len(cur_pre_id)):
            if cur_pre_id[ii]==cur_truth_data:
                cur_pre_id[ii]=1
            else:
                cur_pre_id[ii]=0
    camera_traj_gps,_=merge_each(cmaera_traj_gps,4,total_cand_length)
    camera_traj_time,_=merge_int(camera_traj_time,1,total_cand_length)
    camera_grid_gps,_=merge_each(camera_grid_gps,4,total_cand_length)
    weather_data,_=merge_int(weather_data,1,total_cand_length)
    node_emb,_=merge_each(node_emb,128,total_cand_length)
    isweekend,_=merge_each(isweekend,1,total_cand_length)
    # total_pre_id,_=merge_each(total_pre_id,1,total_cand_length)
    total_pre_id, _ = merge_true_data(total_pre_id, 1)
    # new_truth_data=torch.tensor(total_pre_id)
    # total_cand_length=torch.tensor(total_cand_length)
    return camera_traj_gps, camera_traj_time,\
     camera_grid_gps,\
     total_pre_id,\
     weather_data, node_emb,new_candidate_region,isweekend,total_cand_length
def collate_fn_test(data):
    (cmaera_traj_gps, camera_traj_time,
     camera_traj_node, camera_grid_gps,
     candat_region, truth_data, weather_data,
     node_emb, candidate_region_grid, isweekend) = zip(*data)
    new_candidate_region, total_pre_id = cal_tfidf(candidate_region_grid, cmaera_traj_gps)
    # new_candidate_region=torch.tensor(new_candidate_region)
    new_candidate_region, total_cand_length = merge(new_candidate_region)
    camera_traj_gps, _ = merge_each(cmaera_traj_gps, 4, total_cand_length)
    camera_traj_time, _ = merge_int(camera_traj_time, 1, total_cand_length)
    camera_grid_gps, _ = merge_each(camera_grid_gps, 4, total_cand_length)
    weather_data, _ = merge_int(weather_data, 1, total_cand_length)
    node_emb, _ = merge_each(node_emb, 128, total_cand_length)
    isweekend, _ = merge_each(isweekend, 1, total_cand_length)
    # total_pre_id, _ = merge_each(total_pre_id, 1, total_cand_length)
    pre_id_list=total_pre_id
    return camera_traj_gps, camera_traj_time,\
     camera_grid_gps,\
     weather_data, node_emb,new_candidate_region,isweekend,total_cand_length,pre_id_list,\
     truth_data
if __name__ == '__main__':
    input_data=json.load(open("./data/raw/train/raw_train.json","r"))
    with open("./data/road_network/grid_map.json","r") as f:
        grid_map=json.load(f)
        f.close()
    with open("./data/road_network/node_map.json","r") as f:
        node_map=json.load(f)
        f.close()
    with open("./data/weather/weather_list.json","r") as f:
        weather_data=json.load(f)
        f.close()
    node_emb=np.load("./data/road_network/grid_emb.npy",allow_pickle=True)
    get_data(input_data,grid_map,node_map,weather_data,node_emb)