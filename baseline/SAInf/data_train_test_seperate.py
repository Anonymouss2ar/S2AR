#训练测试6：4
import json
import random


def seperate_data(input_data):
    tran_ratio=0.6
    test_ratio=0.4
    total_key=list(input_data.keys())
    random.shuffle(total_key)
    total_len=len(total_key)
    train_len=int(total_len*tran_ratio)
    test_len=total_len-train_len
    train_id_list=total_key[:train_len]
    test_id_list=total_key[train_len:]
    train_dict={}
    test_dict={}
    for i in range(len(train_id_list)):
        cur_id=train_id_list[i]
        train_dict[i]=input_data[cur_id]
    for j in range(len(test_id_list)):
        cur_test_id=test_id_list[j]
        test_dict[j]=input_data[cur_test_id]
    return train_dict,test_dict,train_id_list,test_id_list
def data_refilter(input_data):
    new_traj={}
    total_key=input_data.keys()
    for each_key in total_key:
        cur_data=input_data[each_key]
        cur_time_stamp=cur_data['timestamp']
        cur_gps=cur_data['gps']
        cur_stay_area=cur_data['stay_area']
        cur_node=cur_data['node']
        car_id=cur_data['id']
        cur_candate_ragion=cur_data['candate_ragion']
        org_list=[i for i in range(len(cur_gps))]
        new_id_list=[]
        new_node_list=[]
        new_candate_ragi_list=[]
        new_stay_area_list=[]
        new_timestamp_list=[]
        new_gps_list=[]
        for j in range(len(cur_candate_ragion)):
            if len(cur_candate_ragion[j])>1:
                new_id_list.append(org_list[j])
        for j in range(len(new_id_list)):#更新list
            cur_new_idx=new_id_list[j]
            new_node_list.append(cur_node[cur_new_idx])
            new_candate_ragi_list.append(cur_candate_ragion[cur_new_idx])
            new_gps_list.append(cur_gps[cur_new_idx])
            new_timestamp_list.append(cur_time_stamp[cur_new_idx])
        for j in range(len(cur_stay_area)):
            tmp_stay_area=cur_stay_area[j]
            if tmp_stay_area[0] in new_id_list:
                new_stay_area_idx=new_id_list.index(tmp_stay_area[0])
                tmp_stay_area[0]=new_stay_area_idx
                new_stay_area_list.append(tmp_stay_area)
            else:
                new_stay_area_idx=len(new_id_list)-1
                tmp_stay_area[0] = new_stay_area_idx
                new_stay_area_list.append(tmp_stay_area)
        if len(new_timestamp_list)>1:
            new_traj[each_key]={}
            new_traj[each_key]['stay_area']=new_stay_area_list
            new_traj[each_key]['node']=new_node_list
            new_traj[each_key]['candate_ragion']=new_candate_ragi_list
            new_traj[each_key]['timestamp']=new_timestamp_list
            new_traj[each_key]['gps']=new_gps_list
            new_traj[each_key]['id']=car_id
    return new_traj
if __name__ == '__main__':
    with open("./data/raw_data/candidate_data.json","r") as f:
        input_data=json.load(f)
        f.close()
    new_traj=data_refilter(input_data)
    train_dict, test_dict, train_id_list, test_id_list=seperate_data(new_traj)
    with open("./data/raw/train/raw_train.json","w") as f:
        json.dump(train_dict, f)
        f.close()
    with open("./data/raw/train/org_train_idx.json","w") as f:
        json.dump(train_id_list, f)
        f.close()
    with open("./data/raw/test/raw_test.json","w") as f:
        json.dump(test_dict, f)
        f.close()
    with open("/data/raw/test/org_test_idx.json","w") as f:
        json.dump(test_id_list,f)
        f.close()