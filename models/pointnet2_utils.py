import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    """
        归一化点云，使用centroid为中心的坐标，半径为1
    """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 球半径，如何理解代码？
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    该函数用来在ball query过程中确定每一个点距离采样点的距离
    函数输入时两组点，N和M分别为前后两组点的个数，C为输入点的通道数（如果是xyz时C=3）
    函数返回的是两组点两两之间的欧氏距离，即N*M的矩阵
    在训练中数据以mini batch的形式输入，batch的数量为B

    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    按照输入的点云数据和索引返回索引的点云数据
    例如points为B*2048*3的点云，idx为[5,666,1000,2000]
    则返回Batch中第5,666,1000,2000个点组成的B*4*3的点云集
    如果idx为一个[V,D1,...,Dn]，则它会被按照idx中的维度结构将其提取成[B,D1,...,Dn,C]

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样。从一个输入点云中按照所需要的点的个数npoints采样出足够多的点
    并且点与其他点的距离最远，
    返回结果为npoins个采样点在原始点云中的索引

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 存储npoint个采样点的索引位置，总共batch size*n个中心点
    distance = torch.ones(B, N).to(device) * 1e10  # 记录某个batch中所有点到某一个点的距离，初始值很大，后面会迭代更新
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 当前最远点，随机初始化batch size个，每个点云一个
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 0~B-1 的数组
    for i in range(npoint):
        centroids[:, i] = farthest  # 第一次用的是随机点，后面用的是计算出的最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 取该centroid的坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 求出所有点到该centroid点的欧氏距离，存在dist矩阵中

        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中的值会慢慢变小
        # 其相当有记录某个batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance

        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    寻找球形邻域中的点。
    输入中radius为球形邻域的半径，nsample为每个邻域中要采样的点
    new_xyz为centroids点的数据，xyz为所有的点云数据
    输出为每个样本的所有球形邻域的nsample个采样点击的索引[B,S,nsample]
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [0]表示只取排序结果，不需要排序结果在-1维上的index
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Sampling + Grouping主要用于将整个点云分散成局部的group,
    对每一个group都可以用PointNet单独地提取局部的全局特征。
    Sampling + Grouping分成了sample_ and group和sample and_group_all两个函数，
    其区别在于sample_and_group_all直接将所有点作为一个group.
    xyz, points为点云数据，这里将坐标和特征分开，为了方便用xyz进行采样
    例如:
    512 = npoint: points sampled in farthest point sampling
    0.2 = radius: search radius in local region
    32 = nsample: how many points in each local region

    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    # 从原点云通过最远点采样挑出的采样点作为new_xyz":
    # 先用farthest point sample函数实现最远点采样得到采样点的索引,
    # 再通过index points将这些点的从原始点中挑出来，作为new_xyz
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)

    # idx:[B, npoint,nsample]，代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # group到的点减去中心点

    # 如果每个点上有新的特征的维度，则拼接新的特征与旧的特征，否则直接返回旧的特征。（用于拼接点特征数据和点坐标数据）
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    PointNetSetAbstraction类实现普通的Set Abstraction:
    首先通过sample_and_group的操作形成局部group,
    然后对局部group中的每一个点做MLP操作，最后进行局部的最大池化，得到局部的全局特征。
    例如: npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128,128,256], group_all=False
        128 = npoint: points sampled in farthest point sampling
        0.4 = radius: search radius in local region
        64 = nsample: how many points in each local region
        [128,128 ,256] = output size for MLP on each point
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        # 以下是pointnet操作:
        # 对局部group中的每一个点做MLP操作:
        # 利用1x1的2d的卷积相当于把每个group当成一个通道，共npoint个通道，
        # 对[C + D, nsample]的维度上做逐像素的卷积，结果相当于对单个C + D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 最后进行局部的最大池化，得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    PointNetSetAbstractionMSG类实现MSG方法的Set Abstraction:这里radius
    list输入的是一个list，例如[0.1,0.2,0.4];
    对于不同的半径做ball query，将不同半径下的点云特征保存在new points list中，最后再拼接到一起。
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        # 例如:128,[0.2,0.4,0.8],[32,64,128],320, [[64,64,128],[128,128,256],[128,128,256]]
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # 最远点采样
        new_points_list = []  # 将不同半径下的点云特征保存在new_points_list
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # 寻找球形邻域的点
            grouped_xyz = index_points(xyz, group_idx)  # 按照输入的点云数据和索引返回索引的点云数据
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # 拼接特征与坐标
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            # 使用pointnet提取点云特征
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)  # 拼接不同半径下的点云特征
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation的实现主要通过线性差值和MLP完成。
    当点的个数只有一个的时候，采用repeat直接复制成N个点;
    当点的个数大于一个的时候，采用线性差值的方式进行上采样，
    拼接上下采样对应点的SA(set abstraction)层的特征，再对拼接后的每一个点都做一个MLP。
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # 当点的个数只有一个的时候，采用repeat直接复制成N个点
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 当点的个数大于一个的时候，采用线性插值的方式进行上采样
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)  # 距离越远的点权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 对于每一个点的群众再做一个全局的归一化
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # 获得插值点

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            # 凭借上下采样前对应点的SA层的特征
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # 对拼接之后的每一个点都进行MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

