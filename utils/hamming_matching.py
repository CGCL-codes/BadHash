import numpy as np
import torch
from tqdm import tqdm

def compute_result(dataloader, net,):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.cuda())).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
# def CalcTopMap(qB, rB, queryL, retrievalL, topk):
#     # qB: {-1,+1}^{mxq}
#     # rB: {-1,+1}^{nxq}
#     # queryL: {0,1}^{mxl}
#     # retrievalL: {0,1}^{nxl}
#     num_query = queryL.shape[0]
#     topkmap = 0
#     # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     for iter in range(num_query):
#         gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
#         hamm = CalcHammingDist(qB[iter, :], rB)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]
#
#         tgnd = gnd[0:topk]
#         tsum = np.sum(tgnd)
#         if tsum == 0:
#             continue
#         count = np.linspace(1, tsum, int(tsum))
#
#         tindex = np.asarray(np.where(tgnd == 1)) + 1.0
#         topkmap_ = np.mean(count / (tindex))
#         # print(topkmap_)
#         topkmap = topkmap + topkmap_
#     topkmap = topkmap / num_query
#     # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     return topkmap


if __name__=='__main__':
    qB = np.array([[1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1]])
    rB = rB = np.array([
        [ 1,-1,-1,-1],
        [-1, 1, 1,-1],
        [ 1, 1, 1,-1],
        [-1,-1, 1, 1],
        [ 1, 1,-1,-1],
        [ 1, 1, 1,-1],
        [-1, 1,-1,-1]])
    queryL = np.array([
        [1,0,0],
        [1,1,0],
        [0,0,1],
    ], dtype=np.int64)
    retrievalL = np.array([
        [0,1,0],
        [1,1,0],
        [1,0,1],
        [0,0,1],
        [0,1,0],
        [0,0,1],
        [1,1,0],
    ], dtype=np.int64)

    topk = 5
    map = CalcMap(qB, rB, queryL, retrievalL)
    topkmap = CalcTopMap(qB, rB, queryL, retrievalL, topk)
    print(map)
    print(topkmap)
