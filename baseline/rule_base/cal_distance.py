import math
import pickle

import torch

EARTH_MEAN_RADIUS_METER = 6371008.7714


def distance(a_lat, a_lng, b_lat, b_lng):
    return haversine_distance(a_lat, a_lng, b_lat, b_lng)


def same_coords(a_lat, a_lng, b_lat, b_lng):
    if a_lat == b_lat and a_lng == b_lng:
        return True
    else:
        return False


def haversine_distance(a_lat, a_lng, b_lat, b_lng):
    if same_coords(a_lat, a_lng, b_lat, b_lng):
        return 0.0
    delta_lat = math.radians(b_lat - a_lat)
    delta_lng = math.radians(b_lng - a_lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a_lat)) * math.cos(
        math.radians(b_lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


def find_first_exceed_max_distance(pt_list_lat,pt_list_lng, cur_idx, max_distance):
    cur_pt_lat = float(pt_list_lat[cur_idx])
    cur_pt_lng = float(pt_list_lng[cur_idx])
    next_idx = cur_idx + 1
    # find all successors whose distance is within MaxStayDist w.r.t. anchor
    while next_idx < len(pt_list_lat):
        next_pt_lat = float(pt_list_lat[next_idx])
        next_pt_lng = float(pt_list_lng[next_idx])
        dist = distance(cur_pt_lat, cur_pt_lng, next_pt_lat, next_pt_lng)
        if dist > max_distance:
            break
        next_idx += 1
    return next_idx


def exceed_max_time(timestamp, cur_idx, next_idx, max_stay_time):
    '''

    :param pt_list:
    :param cur_idx:
    :param next_idx: next idx is the first idx that outside the distance threshold
    :param max_stay_time:
    :return:
    '''
    time_span = timestamp[next_idx - 1] - timestamp[cur_idx]
    # the time span is larger than maxStayTimeInSecond, a stay point is detected
    return time_span > max_stay_time


def euclidean_dist_1(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [m, d]
    Returns:
      dist: pytorch Variable, with shape [m]
    """
    dist = torch.pow(x - y, 2).sum(1, keepdim=True).clamp(min=1e-12).sqrt()

    return dist


def euclidean_dist_2(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist
