> 这个模块提供了使用posenet进行点云分类的部分。

## class get_model(nn.Module)

```python
class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, 
                                    feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat
```

- 该模块定义了分类网络的框架。 ![image-20210907162941303](img/pointnet_cls.assets/image-20210907162941303.png)
- `forward`函数中第一行调用了`pointnet_utils`模块中的全局特征提取部分，`global_feat=True`表示直接返回global feature，不需要与local feature 进行拼接，因为分类不需要每个点的local feature。`feature_transform`表示需要对第一次提取到的feature经过一个T-Net变换。
- 接下来接上三个全连接层，将global feature从1024维降到k维，k为类别数。
- **为什么最后一个全连接层后面不接BN和Relu，而直接接一个Soft Max？**

## class get_loss(torch.nn.Module)

```python
class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
```

- `F.nll_loss(pred, target)`计算的是分类错误。
- `feature_transform_reguliarzer(trans_feat)`计算的是特征转换矩阵的正交Loss。
- 最后的loss为两个loss的总和。