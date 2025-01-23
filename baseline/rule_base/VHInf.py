import json
import logging
import pickle

from utils1 import search_cand_interval, test_method
from cal_distance import distance


def SHInf(traj_data,aoiwithgrid,grid2aoi,grid_map,node2gps,car_aoi_pinlv):
    total_key_list = list(traj_data.keys())
    logging.info("total len: %d", len(total_key_list))
    total_pred_id=[]
    total_truth_list=[]
    iiii=0
    for each_traj_key in total_key_list:
        cur_data = traj_data[each_traj_key]
        cur_gps = cur_data['gps']
        # cur_four_tuple_time = cur_data['four_tuple_time']
        cur_stay_area = cur_data['stay_area']
        cur_timestamp = cur_data['timestamp']
        tmp_gps_pair=[]
        tmp_aoi_truth_idx=[]
        tmp_aoi_truth_id=[]
        tmp_timstamp_pair=[]
        for i in range(len(cur_gps)-1):
            tmp_gps_pair.append([cur_gps[i],cur_gps[i+1]])
            tmp_timstamp_pair.append([cur_timestamp[i],cur_timestamp[i+1]])

        for j in range(len(cur_stay_area)):
            each_stay_list=cur_stay_area[j]
            tmp_aoi_truth_idx.append(each_stay_list[0][0])
            cur_stay_gps=each_stay_list[3]#驻留区域的gps
            grid_id,_,_,_=grid_map.get_cell_by_point(cur_stay_gps[0],cur_stay_gps[1])
            aoi_id=grid2aoi[str(grid_id)]
            tmp_aoi_truth_id.append(aoi_id)
        for k in range(len(tmp_gps_pair)):
            cur_tmp_gps_pair=tmp_gps_pair[k]
            cur_timestamp_pair=tmp_timstamp_pair[k]
            cur_dist=distance(cur_tmp_gps_pair[0][1],cur_tmp_gps_pair[0][0],cur_tmp_gps_pair[1][1],cur_tmp_gps_pair[1][0])
            cur_time_span=cur_timestamp_pair[1]-cur_timestamp_pair[0]
            cur_v=cur_dist/cur_time_span
            if cur_v<3:
                cand_list=search_cand_interval(cur_tmp_gps_pair,aoiwithgrid,cur_dist)
                if k in tmp_aoi_truth_idx:
                    truth_aoi_tmp_idx = tmp_aoi_truth_idx.index(k)
                    cur_aoi_id = tmp_aoi_truth_id[truth_aoi_tmp_idx]
                else:
                    cur_aoi_id=10000
                tmp_paixu_list = []
                if len(cand_list) > 0:
                    for each_cand_area in cand_list:
                        if str(each_cand_area) in car_aoi_pinlv:
                            occur_pinlv = car_aoi_pinlv[str(each_cand_area)]
                        else:
                            occur_pinlv = 0
                        tmp_paixu_list.append([each_cand_area, occur_pinlv])
                    sorted_list = sorted(tmp_paixu_list, key=lambda d: d[1])
                    new_pre_list = []
                    for ijiji in range(len(sorted_list)):
                        new_pre_list.append(sorted_list[ijiji][0])
                    total_pred_id.append(new_pre_list)
                    total_truth_list.append(cur_aoi_id)
        logging.info("finish traj : {%d}", iiii)
        iiii += 1
    with open("./data/VHInf/pred_list.json", "w") as f:
        json.dump(total_pred_id, f)
        f.close()
    with open("./VHInf/truth_id.json", "w") as f:
        json.dump(total_truth_list, f)
        f.close()
    test_method(total_truth_list, total_pred_id, node2gps)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler(
                                './log/VHInf/0120_VHInf.log',
                                mode='w'),
                            logging.StreamHandler()]
                        )

    with open("./data/CSS/test/test_data_tmp.json", "r") as f:
        traj_data = json.load(f)
        f.close()
    with open("./data/AOI/aoi_with_grid.json", "r") as f:
        aoiwithgrid = json.load(f)
        f.close()
    with open("./data/AOI/grid2aoi.json", "r") as f:
        grid2aoi = json.load(f)
        f.close()
    with open("./data/AOI/grid_map.pkl", "rb") as f:
        grid_map = pickle.load(f)
        f.close()
    with open("./data/AOI/aoi2gps_taian.json", "r") as f:
        node2gps = json.load(f)
        f.close()
    with open("./data/AOI/car_aoi_appear_time.json","r") as f:
        car_aoi_pinlv=json.load(f)
        f.close()
    logging.info("finish read data")
    SHInf(traj_data,aoiwithgrid,grid2aoi,grid_map,node2gps,car_aoi_pinlv)
    
