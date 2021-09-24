## def pc_normalize(pc):

```python
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 球半径，如何理解代码？
    pc = pc / m
    return pc
```

- 该函数用于对输入点云（pc）做坐标归一化。公式为：
  $$
  p_{normalize} = \frac{p-p_{mean}}{max {||(p_i-p_{mean})-(p_j-p_{mean})||_2}},\quad i,j=1,2...n
  $$

- 需要注意的是`pc**2`代表的是张量中的每个元素进行平方运算，由于第三行已经对所有点减去了点云均值，相当于将点云的中心点移动到了坐标轴的中心，这样求得的平方代表的是每个点到零点的距离，也就是每个点到点云中心点的距离。取所有点到点云中心点的最远距离作为点云的球半径。

- `pc = pc / m`将所有点的坐标归一化到0~1。

## def square_distance(src, dst)

```python
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
```

- 该函数用于在ball query过程中确定每一个点距离采样点的距离。函数返回src点云中的每一个点与dst点云中的每一个点之间的距离，使用一个`N*M`的矩阵进行表示。返回的矩阵的shape为`B*N*M`，其中第`i`个`N*M`矩阵为src中第`i`个点云矩阵与dst中第`i`个点云矩阵之间的距离矩阵。

- B = batch size，N = src中点的数量，M = dst中点的数量。

- 计算src中的第`i`个二维矩阵与dst中的第`i`个矩阵之间的点与点的平方距离。

- `dist = -2 torch.matmul(src, dst.permute(0, 2, 1))`计算了
  $$
  src^T * dst = x_n*x_m + y_n*y_m + z_n*z_m
  $$

- 4-6行计算得到了一个`M*N`的矩阵，其每个元素为src点云中的一个点$p_n=(x_n,y_n,z_n)$与$p_m=(x_m,y_m,z_m)$两个点的欧式距离（省去了开平方）。
  $$
  (x_n - x_m)^2 + (y_n - y_m)^2 + (z_n - z_m)^2=\\x_n^2+y_n^2+y_n^2+z_n^2+x_m^2+y_m^2+z_m^2-2x_nx_m-2y_ny_m+z_nz_m
  $$

## def index_points(points, idx)

```python
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device) \
    			   .view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
```

- 按照输入的点云数据points（B,N,C），和点的索引idx（B, S），返回点云中指定index的点组成的子点云。

## def farthest_point_sample(xyz, npoint)

```python
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
```

- 最远点采样函数。从一个输入点云中按照最远距离原则采样npoint个点。所谓最远距离原则就是每次在选择一个新的采样点时，选择点云中距离当前已经被选取的所有采样点最远的点，这里的距离使用的是一个点距离所有采样点的最短平方欧氏距离。

- 几个比较重要的变量：

  - centroids：存储`B*npoint`个centroid的**index**，每个点根据最远距离原则进行采样。最后的for循环的每一层都会填充一列，即为每个点云采样一个新的cnetroid。
  - distance：shape为`(B*N)`，存储所有点距离所属点云的所有采样点的最近距离。在每一轮for循环里会计算所有点到新采样点的距离，如果新距离小于distance中存储的距离，就用更小的新距离替换掉 。
  - farthest：shape为`(B,)`，记录距离当前所有采样点最远的点的**index**。其在一开始被随机为0~N-1的整数，表示在点云中随机选择一个点作为第一个centroid。其在for循环的最后更新为distance张量中每个点云的最大值的index。

- 该函数的伪代码如下：

  ```python
  def farthest_point_sample(xyz, npoint):
      # xyz为输入的点云，shape为(B,N,3)，npoint为想要采样的点的数量
      B,N ← batch size, 每个点云中包含的点数
      centroid ← B*npoint大小的0矩阵
      distance ← B*N大小的矩阵，初始化每个元素为无穷大
      farthest ← 长度为B的向量，每个元素为0~N-1的随机数
      batch_indices ← 长度为B的顺序向量，其中的值依次为0,1,2......
      for i from 0 to npoint:
          centroids的第i列 ← farthest（距离当前所有采样的距离最大的点的index，总共B个）
          centroid ← farthest指定的index的点的坐标，总共B个
          dist ← 所有点到所属的centroid的距离，B*N大小的矩阵
          mask ← B*N大小的矩阵，对应位置的dist<distance，则为1
          对于mask为1的位置，使用dist在该位置的值更新distance在该位置的值
          farthest ← distance每一行的最大值（代表该点云中距离所有采样点的最大距离）
  ```

## def query_ball_point(radius, nsample, xyz, new_xyz)

```python
def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N) \ 
    			.repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
```

- 该函数用于寻找点云xyz在以new_xyz中的点为圆心，radius为半径的球形区域中的nsample个点。
- 重要变量介绍
  - `radius`：float，球邻域半径
  - `nsample`：int，每个球邻域采样的点数
  - `xyz`：shape = (B,N,3)，输入的B个点云
  - `new_xyz`：shape = (B,S,3)，每个点云有S个球邻域需要查询。
- 第七行，`sqrdists`为一个shape为(B,S,N)的张量，记录了每个`centroid`与所属点云的其他点的距离。
- 第六行，`group_idx`被初始化为一个`B*S*N`的矩阵（与`sqrdists`形状相同），其包含`B*S`个0~N-1的顺序向量；第八行，将`sqrdists`中距离小于半径的位置的`group_idx`的值更新为`N`；第九行，将`group_idx`从小到大排序，将那些距离centroid大于半径的点排在后面，然后取前`nsample`个点作为球邻域的采样点。
- 上面这种采样方式假定每个球邻域中的点不少于`nsample`个点，否则就会取到半径外的点。第10~13行就是为了解决这个问题，解决方法是将采样到的半径外的点替换成距离centroid最近的点。
- 第十行，`group_first`是一个`B*S*nsample`的张量，B个二维矩阵包含S个centroid，某一列中的元素是重复的，均为距离 centroid最近的点的index。
- 第十一行得到采样到半径外的点的index。
- 第十二行将这些半径外的点更新为距离centroid最近的点的index。

## def sample_and_group()

```python
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # group到的点减去中心点
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
```

- Sampling + Grouping主要用于将整个点云分散成局部的group，对每一个group都可以用PointNet单独地提取局部的全局特征。
- 一些重要的参数介绍：
  - npoint：在一个点云中需要采样npoint个最远点，也表示一个点云分为多少组。
  - radius：球邻域半径
  - nsample：每个球邻域采样的点数，作为一个group。
  - xyz：[B, N, 3]，输入点云的位置信息
  - points：[B, N, D]，输入点云的特征信息。一个完整的点云应当是(B,N,3+D)的。
  - new_xyz：[B, npoint, nsample, 3]，将点云分组后的张量，每个点云分为npoint组，每组里包含nsample个点。
  - new_points：[B, npoint, nsample, 3+D]，比上面的点云多了特征。
  - idx：[B, npoint,nsample]，代表npoint个球形区域中每个区域的nsample个采样点的索引。
- 第四行得到`B*npoint`个centroid的index，这里可以不用管C，因为它必是3，所以可以直接用3代替就好。
- 第五行得到这些centroid的坐标。第六行和第七行在这些centroid的球邻域中采样nsample个点，已经得到了[B, npoint, nsample, C]的张量了。
- 第八行将每个group的点减去对应的centroid坐标，将每个group的中心移动到坐标原点。这样就已经将原来的整个点云分成很多组点云，并且将这些点云都移动到坐标系原点，这样每个group都可以使用pointnet进行全局特征提取了。
- 由于上面的group操作只关心了点的坐标，九到十三行是判断是否要考虑点的其他特征，如果考虑，那就用sample的index找到对应的特征行，拼接到group到的点云上去。

