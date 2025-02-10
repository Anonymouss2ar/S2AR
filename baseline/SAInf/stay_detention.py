import json
from collections import Counter

from matplotlib import pyplot as plt

from stay_area.util.cal_distance import distance


def stay_event_detention(input_data):
    stay_speed_list=[]
    unstay_speed_list=[]
    for i in range(len(input_data)):
        cur_data=input_data[str(i)]
        cur_timestamp=cur_data['timestamp']
        cur_gps=cur_data['gps']
        cur_stay_area=cur_data['stay_area']
        stay_idx=[]
        for j in range(len(cur_stay_area)):
            tmp_cur_stay_area=cur_stay_area[j]
            stay_idx.append(tmp_cur_stay_area[0][0])
        for j in range(0,len(cur_gps)-1):
            tmp_gps=cur_gps[j]
            dist=distance(tmp_gps[1],tmp_gps[0],cur_gps[j+1][1],cur_gps[j+1][0])
            time_span=cur_timestamp[j+1]-cur_timestamp[j]
            tmp_speed=float(dist/time_span)
            if tmp_speed<27.778 :
                if j in stay_idx:
                    stay_speed_list.append(tmp_speed)
                else:
                    unstay_speed_list.append(tmp_speed)
        if i %5000==0:
            print("finish :", i)
    return stay_speed_list,unstay_speed_list
def new_stay_area_detect(stay_speed_list,unstay_speed_list):
    stay_speed_quanter = quantize(stay_speed_list)
    unstay_speed_quanter = quantize(unstay_speed_list)
    max_diff_index, max_diff, final_speed = max_cdf_difference(stay_speed_quanter, unstay_speed_quanter)
    return final_speed
def empirical_cdf(data):
    sorted_data = sorted(data)
    count = Counter(sorted_data)
    total_elements = sum(count.values())
    cdf = []
    cumulative_count = 0
    frequency_list=[]
    for value, frequency in count.items():
        cumulative_count += frequency
        cdf.append(cumulative_count / total_elements)
        frequency_list.append(frequency)
    return list(count.keys()), cdf,frequency_list
    # return sorted_data, cdf


def max_cdf_difference(list1, list2):
    # 计算两个列表的CDF
    sorted_list1, cdf1,frequence1 = empirical_cdf(list1)
    sorted_list2, cdf2,frequence2 = empirical_cdf(list2)
    plt.plot(sorted_list1,cdf1,linewidth=2,c="orange")
    plt.plot(sorted_list2,cdf2,c="blue")
    plt.show()
    plt.bar(sorted_list1,frequence1,color="orange",label="stay area",alpha=0.8)
    plt.bar(sorted_list2,frequence2,color="blue",label="unstay area",alpha=0.8)
    plt.legend()
    plt.show()
    # 找出两个CDF长度的最大值，以确保可以比较所有元素
    max_length = min(len(cdf1), len(cdf2))

    # 初始化最大差值和对应的下标
    max_diff = 0
    max_diff_index = -1

    # 比较两个CDF的差值
    diss_list=[]
    for i in range(max_length):
        # 确保不会访问不存在的CDF值
        val1 = cdf1[i]
        val2 = cdf2[i]
        # 计算差值
        diff = abs(val1 - val2)
        diss_list.append(diff)
    max_diff=max(diss_list)
    max_diff_index=diss_list.index(max_diff)
    return max_diff_index, max_diff,sorted_list1[max_diff_index]
def get_canda_stay_event(input_data,threoud_speed):#筛选出包含驻留事件的摄像头对，减少候选摄像头对的数量
    new_traj={}
    total_keys=input_data.keys()
    for i,each_key in enumerate(total_keys):
        each_data=input_data[each_key]
        cur_timestamp = each_data['timestamp']
        cur_gps = each_data['gps']
        cur_stay_area = each_data['stay_area']
        cur_node=each_data['node']
        cur_id=each_data['id']
        new_camera_id=[]
        for j in range(0,len(cur_gps)-1):
            tmp_gps=cur_gps[j]
            dist=distance(tmp_gps[1],tmp_gps[0],cur_gps[j+1][1],cur_gps[j+1][0])
            time_span=cur_timestamp[j+1]-cur_timestamp[j]
            tmp_speed=float(dist/time_span)
            if tmp_speed<threoud_speed:
                new_camera_id.append([j,j+1])
        new_gps=[]
        new_time_stamp=[]
        new_node=[]
        for j in range(len(new_camera_id)):
            start_idx=new_camera_id[j][0]
            end_idx=new_camera_id[j][1]
            start_gps=cur_gps[start_idx]
            end_gps=cur_gps[end_idx]
            start_timestamp=cur_timestamp[start_idx]
            end_timestamp=cur_timestamp[end_idx]
            start_node=cur_node[start_idx]
            end_node=cur_node[end_idx]
            new_gps.append([start_gps,end_gps])
            new_time_stamp.append([start_timestamp,end_timestamp])
            new_node.append([start_node,end_node])
        new_stay_area=[]
        for j in range(len(cur_stay_area)):
            tmp_stay_area=cur_stay_area[j]
            tmp_stay_area_id=tmp_stay_area[0]
            a=0
            for mm in range(len(new_camera_id)):
                start_idx=new_camera_id[mm][0]
                if start_idx == tmp_stay_area_id[0]:
                    # if start_idx<len(new_camera_id)-2:
                    cur_stay_area[j][0]=mm
                    new_stay_area.append([cur_stay_area[j][0],cur_stay_area[j][1],cur_stay_area[j][2],cur_stay_area[j][3],cur_stay_area[j][5],cur_stay_area[j][4]])
                    a+=1
        new_traj[i]={}
        new_traj[i]['timestamp']=new_time_stamp
        new_traj[i]['gps']=new_gps
        new_traj[i]['stay_area']=new_stay_area
        new_traj[i]['node']=new_node
        new_traj[i]['id']=cur_id
    return new_traj

def quantize(data, interval=0.015):
    # 量化数据到指定的间隔
    return [round(x / interval) * interval for x in data]
if __name__ == '__main__':
    with open("./data/raw/raw_camera.json","r") as f:
        input_data=json.load(f)
        f.close()
    stay_speed_list,unstay_speed_list=stay_event_detention(input_data)
    final_speed=new_stay_area_detect(stay_speed_list, unstay_speed_list)
    new_traj=get_canda_stay_event(input_data,final_speed)
    with open("./data/stay_event/stay_event_data.json","w") as f:
        json.dump(new_traj,f)
        f.close()