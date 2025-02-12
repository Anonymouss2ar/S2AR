import json
from multiprocessing import Pool

import numpy as np

from stay_area.util.cal_distance import distance
def get_ratio(input_data):
    total_ratio=[]
    for i in range(len(input_data)):
        cur_data = input_data[str(i)]
        cur_gps=cur_data['gps']
        cur_stay_area = cur_data['stay_area']
        for j in range(len(cur_stay_area)):
            tmp_cur_stay_area=cur_stay_area[j]
            start_idx=tmp_cur_stay_area[0]
            stay_point_gps=tmp_cur_stay_area[3]
            start_camera_gps=cur_gps[start_idx][0]
            end_camera_gps=cur_gps[start_idx][1]
            cur_2c=distance(start_camera_gps[1],start_camera_gps[0],end_camera_gps[1],end_camera_gps[0])
            cur_2a=distance(start_camera_gps[1],start_camera_gps[0],stay_point_gps[1],stay_point_gps[0])+distance(end_camera_gps[1],end_camera_gps[0],stay_point_gps[1],stay_point_gps[0])
            total_ratio.append(float(cur_2a/cur_2c))
        print("finish : ",i)
    # 计算经验累积概率
    cumulative_probabilities = np.cumsum(sorted(total_ratio)) / len(total_ratio)
    a_over_c_at_95_percentile = np.interp(0.95, cumulative_probabilities, total_ratio)
    return a_over_c_at_95_percentile
def multiprocess_cand(cell,hang,cur_gps,ratio,i,cur_stay_idx):
    each_traj_candate=[]
    for j in range(len(cur_gps)):
        tmp_flag=0
        cur_gps_list=cur_gps[j]
        pre_gps=cur_gps_list[0]
        gps_now=cur_gps_list[1]
        cur_2c = distance(pre_gps[1], pre_gps[0], gps_now[1], gps_now[0])
        cur_2a=ratio*cur_2c
        tmp_candiate = []
        for cell_h in hang:
            cur_cell_lie = list(cell[cell_h].keys())
            for cell_lie in cur_cell_lie:
                cur_cell_gps = cell[cell_h][cell_lie][1]
                cal_a = distance(pre_gps[1], pre_gps[0], cur_cell_gps[1], cur_cell_gps[0])
                if cal_a < cur_2a:
                    cal_2a = cal_a + distance(gps_now[1], gps_now[0], cur_cell_gps[1], cur_cell_gps[0])
                    if cal_2a <= cur_2a:
                        tmp_candiate.append([cell_h, cell_lie])
        for mm in range(len(cur_stay_idx)):
            if cur_stay_idx[mm] in tmp_candiate:
                tmp_flag+=1
        if tmp_flag!=0:
            each_traj_candate.append(tmp_candiate)
    print("finished ", i)
    return [each_traj_candate,1]
def generate_cand_ragion(ratio,input_data,cell):
    hang=list(cell.keys())
    pool = Pool(processes=20)
    multi_result = []
    for i in range(len(input_data)):
        print("start :",i)
        cur_data = input_data[str(i)]
        cur_gps=cur_data['gps']
        cur_stay_area=cur_data['stay_area']
        cur_h_l=[[str(cur_stay_area[mmm][4]),str(cur_stay_area[mmm][5])] for mmm in range(len(cur_stay_area))]
        tmp = pool.apply_async(multiprocess_cand,
                               args=(cell, hang, cur_gps, ratio,i,cur_h_l))
        multi_result.append(tmp)
    pool.close()
    pool.join()
    for idx,each_tmp in enumerate(multi_result):
        tmp_result = each_tmp.get()
        input_data[str(idx)]["candate_ragion"]=tmp_result[0]
    return input_data

if __name__ == '__main__':
    with open("./data/stay_event/stay_event_data.json","r") as f:
        input_data=json.load(f)
        f.close()
    ratio=get_ratio(input_data)
    print(ratio)
    with open("./data/road_network/cell_data_with_poi.json","r") as f:
        cell_data=json.load(f)
        f.close()
    final_data=generate_cand_ragion(ratio,input_data,cell_data)
    with open("./data/raw_data/candidate_data.json","w") as f:
        json.dump(final_data,f)
        f.close()
