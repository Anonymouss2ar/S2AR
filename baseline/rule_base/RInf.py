import json
import logging
import pickle
import random
import sys

from utils1 import search_candidate_mid, test_method
# sys.path.append('/extern1/dxy/all/')

def RInf(traj_data,aoiwithgrid,grid2aoi,grid_map,node2gps):
    total_key_list = list(traj_data.keys())
    logging.info("total len: %d",len(total_key_list))
    total_pred_id=[]
    total_truth_list=[]
    iiii=0
    for each_traj_key in total_key_list:
        cur_data = traj_data[each_traj_key]
        cur_gps = cur_data['gps']
        # cur_four_tuple_time = cur_data['four_tuple_time']
        cur_stay_area = cur_data['stay_area']
        tmp_gps_pair=[]
        tmp_aoi_truth_idx=[]
        tmp_aoi_truth_id=[]
        for i in range(len(cur_gps)-1):
            tmp_gps_pair.append([cur_gps[i],cur_gps[i+1]])
        for j in range(len(cur_stay_area)):
            each_stay_list=cur_stay_area[j]
            tmp_aoi_truth_idx.append(each_stay_list[0][0])
            cur_stay_gps=each_stay_list[3]#驻留区域的gps
            grid_id,_,_,_=grid_map.get_cell_by_point(cur_stay_gps[0],cur_stay_gps[1])
            aoi_id=grid2aoi[str(grid_id)]
            tmp_aoi_truth_id.append(aoi_id)
            # input_gps=[cur_gps[each_stay_list[0][0]],cur_gps[each_stay_list[0][1]]]
        for idx,each_camera_pair in enumerate(tmp_gps_pair):
            if idx in tmp_aoi_truth_idx:
                truth_aoi_tmp_idx=tmp_aoi_truth_idx.index(idx)
                cur_aoi_id=tmp_aoi_truth_id[truth_aoi_tmp_idx]
                cand_list=search_candidate_mid(each_camera_pair,aoiwithgrid,grid2aoi)
                random.shuffle(cand_list)
                if len(cand_list)>2:
                    total_pred_id.append(cand_list)
                    total_truth_list.append(cur_aoi_id)
            else:
                cand_list = search_candidate_mid(each_camera_pair, aoiwithgrid, grid2aoi)
                random.shuffle(cand_list)
                if len(cand_list) > 2:
                    total_pred_id.append(cand_list)
                    cur_aoi_id=cand_list[-1]
                    total_truth_list.append(cur_aoi_id)
        logging.info("finish traj : {%d}",iiii)
        iiii+=1
    with open("./data/RInf/pred_list.json","w") as f:
        json.dump(total_pred_id,f)
        f.close()
    with open("./RInf/truth_id.json","w") as f:
        json.dump(total_truth_list,f)
        f.close()
    test_method(total_truth_list,total_pred_id,node2gps)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler(
                                './log/RInf/0121_RInf_new.log',
                                mode='w'),
                            logging.StreamHandler()]
                        )
    with open("./data/aoi2gps_taian.json","r") as f:
        node2gps=json.load(f)
        f.close()
    with open("./CSS/test/test_data_tmp.json", "r") as f:
        traj_data = json.load(f)
        f.close()
    with open("./AOI/aoi_with_grid.json", "r") as f:
        aoiwithgrid = json.load(f)
        f.close()
    with open("./AOI/grid2aoi.json", "r") as f:
        grid2aoi = json.load(f)
        f.close()
    with open("./AOI/grid_map.pkl", "rb") as f:
        grid_map = pickle.load(f)
        f.close()
    logging.info("finish read data")
    RInf(traj_data, aoiwithgrid, grid2aoi, grid_map, node2gps)
