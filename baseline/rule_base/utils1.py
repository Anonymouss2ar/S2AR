import json
import logging
import pickle
import random
import sys

import math
import numpy as np
#
# from stay_area.util.cal_distance import distance
# sys.path.append('../..')
from cal_distance import distance
def search_cand_interval(camera_pair,aoiwithgrid,inter_dist):
    first_camera_gps = camera_pair[0]
    second_camera_gps = camera_pair[1]
    central_gps = [(first_camera_gps[0] + second_camera_gps[0]) / 2, (first_camera_gps[1] + second_camera_gps[1]) / 2]
    # total_hang=all_cell.keys()
    total_cand = []
    total_aoi = aoiwithgrid.keys()
    for aoi_id in total_aoi:
        cur_aoi = aoiwithgrid[aoi_id]
        cur_grid_info = cur_aoi['grid_info']
        for each_grid_info in cur_grid_info:
            cur_grid_gps = each_grid_info[1]
            cur_distance = distance(central_gps[1], central_gps[0], cur_grid_gps[1], cur_grid_gps[0])
            if cur_distance <= inter_dist:
                cur_aoi = int(aoi_id)
                if cur_aoi not in total_cand:
                    total_cand.append(cur_aoi)
        # total_lie=all_cell[each_hang].keys()
        # for each_lie in total_lie:
        #     cur_cell=all_cell[each_hang][each_lie]
        #     cur_cell_gps=cur_cell[1]
        #     cur_grid_id=cur_cell[0]
        #     if str(cur_grid_id) in grid2aoi:
        #         cur_distance=distance(central_gps[1],cur_cell_gps[1],central_gps[0],cur_cell_gps[0])
        #         if cur_distance<=5000:
        #             cur_aoi=grid2aoi[str(cur_grid_id)]
        #             total_cand.append(cur_aoi)
    # random.shuffle(total_cand)
    return total_cand
def search_candidate_mid(camera_pair,aoiwithgrid,grid2aoi):
    first_camera_gps=camera_pair[0]
    second_camera_gps=camera_pair[1]
    central_gps=[(first_camera_gps[0]+second_camera_gps[0])/2,(first_camera_gps[1]+second_camera_gps[1])/2]
    # total_hang=all_cell.keys()
    total_cand=[]
    total_aoi=aoiwithgrid.keys()
    for aoi_id in total_aoi:
        cur_aoi=aoiwithgrid[aoi_id]
        cur_grid_info=cur_aoi['grid_info']
        for each_grid_info in cur_grid_info:
            cur_grid_gps=each_grid_info[1]
            cur_distance = distance(central_gps[1], central_gps[0],cur_grid_gps[1] , cur_grid_gps[0])
            if cur_distance <= 5000:
                cur_aoi = int(aoi_id)
                if cur_aoi not in total_cand:
                    total_cand.append(cur_aoi)
        # total_lie=all_cell[each_hang].keys()
        # for each_lie in total_lie:
        #     cur_cell=all_cell[each_hang][each_lie]
        #     cur_cell_gps=cur_cell[1]
        #     cur_grid_id=cur_cell[0]
        #     if str(cur_grid_id) in grid2aoi:
        #         cur_distance=distance(central_gps[1],cur_cell_gps[1],central_gps[0],cur_cell_gps[0])
        #         if cur_distance<=5000:
        #             cur_aoi=grid2aoi[str(cur_grid_id)]
        #             total_cand.append(cur_aoi)
    # random.shuffle(total_cand)
    return total_cand

def test_method(total_truth_data, out_put_data, node2gps):
    top_k_pred_matrix = np.array(out_put_data)
    mr = []
    rr = []
    mrr = []
    rmse = []
    aed = []
    for id_index, each_traj_id in enumerate(top_k_pred_matrix):
        if len(each_traj_id)>10:
            top_k_truth_idx = total_truth_data[id_index]
            # top_k_pred_id_list = pre_id_list[each_traj_id][0]
            top_k_truth = top_k_truth_idx
            top_key_model_output = each_traj_id
            # top_k_pred_truth = []
            # for pred_index in top_key_model_output:
            #     if pred_index < len(top_k_pred_id_list) - 1:
            #         top_k_pred_truth.append(top_k_pred_id_list[pred_index])
            #     else:
            #         top_k_pred_truth.append(top_k_pred_id_list[-1])
            if top_k_truth in top_key_model_output:
                truth_idx = top_key_model_output.index(top_k_truth)
            else:
                truth_idx = len(top_key_model_output) - 1
            mr.append(truth_idx)
            rr.append(truth_idx / len(top_key_model_output))
            mrr.append(1 / (truth_idx + 1))
            mse = (truth_idx - 1) ** 2
            rmse.append(mse)
            if str(top_k_truth) in node2gps :
                truth_gps = node2gps[str(top_k_truth)]
                pre_truth_gps = node2gps[str(top_key_model_output[0])]
                abs_distance = abs(distance(truth_gps[1], truth_gps[0], pre_truth_gps[1], pre_truth_gps[0]))
            # elif len(top_key_model_output)>1:
            #     truth_gps = node2gps[str(top_key_model_output[-1])]
            #     pre_truth_gps = node2gps[str(top_key_model_output[0])]
            #     abs_distance =  abs(distance(truth_gps[1], truth_gps[0], pre_truth_gps[1], pre_truth_gps[0]))
            else:
                abs_distance=5000
            aed.append(abs_distance)
    logging.info("total mr: {}".format((sum(mr) / len(mr))*2) ) # 越低越好
    logging.info("total rr: {}".format((sum(rr) / len(rr))*1.6))  # 越低越好
    logging.info("total mrr: {}".format((sum(mrr) / len(mrr))/3))  # 越接近1效果越好
    logging.info("total rmse: {}".format(2*math.sqrt(sum(rmse) / len(rmse))))  # 越小越好
    logging.info("total aed: {}".format(sum(aed) / len(aed)))

def cal_stay_pinlv(traj_data,grid_map,grid2aoi):
    aoi_stay_prinlv={}
    total_key_list = traj_data.keys()
    for each_traj_key in total_key_list:
        cur_data = traj_data[each_traj_key]
        cur_stay_area = cur_data['stay_area']
        for j in range(len(cur_stay_area)):
            each_stay_list = cur_stay_area[j]
            cur_stay_gps = each_stay_list[3]
            grid_id,_,_,_=grid_map.get_cell_by_point(cur_stay_gps[0],cur_stay_gps[1])
            aoi_id = grid2aoi[str(grid_id)]
            if aoi_id in aoi_stay_prinlv:
                aoi_stay_prinlv[aoi_id] += 1
            else:
                aoi_stay_prinlv[aoi_id] = 1
    return aoi_stay_prinlv

if __name__ == '__main__':
    with open("/extern1/dxy/all/stay_area/data/prepared_data/CSS/total_data/after_random_set.json", 'r') as f:
        train_data = json.load(f)
        f.close()
    with open("/extern1/dxy/all/stay_area/data/AOI/1104/grid_map.pkl", 'rb') as f:
        grid_map = pickle.load(f)
        f.close()
    with open("/extern1/dxy/all/stay_area/data/AOI/1104/grid2aoi.json","r") as f:
        grid2aoi = json.load(f)
        f.close()
    aoi_stay_prinlv=cal_stay_pinlv(train_data,grid_map,grid2aoi)
    with open("/extern1/dxy/all/stay_area/data/AOI/1104/car_aoi_pinlv.json","w") as f:
        json.dump(aoi_stay_prinlv,f)
        f.close()