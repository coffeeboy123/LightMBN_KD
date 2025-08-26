from torch import nn

class BNNeck(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)

        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck3(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(feat_dim)

        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        # before_neck = x.squeeze(dim=3).squeeze(dim=2)
        # after_neck = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)
        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck_qat(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck_qat, self).__init__()
        self.return_f = return_f
        # 4D 텐서 (N, C, 1, 1)을 받는 BatchNorm2d 사용
        self.bn = nn.BatchNorm2d(input_dim)
        # BN의 bias는 학습하지 않음
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        # x: (N, input_dim, 1, 1)로 가정합니다.
        before_neck = x  # (N, C, 1, 1)
        after_neck = self.bn(before_neck)  # BatchNorm2d는 4D 입력을 받으므로 (N, C, 1, 1)
        # flatten 후 (N, C) 형태로 변환
        after_neck_flat = after_neck.view(after_neck.size(0), -1)
        if self.return_f:
            score = self.classifier(after_neck_flat)
            # 필요에 따라 flatten 전의 feature도 (N, C) 형태로 반환
            pre_bn = before_neck.view(before_neck.size(0), -1)
            return after_neck_flat, score, pre_bn
        else:
            out = self.classifier(after_neck_flat)
            return out

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck3_qat(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3_qat, self).__init__()
        self.return_f = return_f
        # 1x1 Convolution으로 차원 축소: (N, input_dim, H, W) -> (N, feat_dim, H, W)
        self.reduction = nn.Conv2d(input_dim, feat_dim, 1, bias=False)
        # 4D 텐서를 받는 BatchNorm2d 사용 (BN2d는 (N, C, H, W)를 기대)
        self.bn = nn.BatchNorm2d(feat_dim)
        # BN의 bias는 학습하지 않음.
        self.bn.bias.requires_grad_(False)
        # 분류기: 최종 출력 피처는 feat_dim 차원 (flatten 후 적용)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        # x: (N, input_dim, H, W)
        x = self.reduction(x)  # (N, feat_dim, H, W)
        # 만약 spatial 차원이 1이 아니라면 pooling 적용하여 (N, feat_dim, 1, 1)로 변환
        if x.size(2) != 1 or x.size(3) != 1:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        # BatchNorm2d 적용 (입력은 4D: (N, feat_dim, 1, 1))
        bn_out = self.bn(x)
        # flatten하여 (N, feat_dim) 형태로 변환
        feat = bn_out.view(bn_out.size(0), -1)
        if self.return_f:
            score = self.classifier(feat)
            # 원본 (풀링 전) 텐서도 flatten해서 반환할 수 있음
            pre_bn = x.view(x.size(0), -1)
            return feat, score, pre_bn
        else:
            out = self.classifier(feat)
            return out

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x.squeeze(3).squeeze(2))
        if self.return_f:
            f = x
            x = self.classifier(x)
            return f, x, f
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            # For old pytorch, you may use kaiming_normal.
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)




class BNNeck_neck(nn.Module):
    def __init__(self, input_dim, return_f=False):
        super(BNNeck_neck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(self.weights_init_kaiming)

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)


        return after_neck, before_neck

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck_classifier(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck_classifier, self).__init__()
        self.return_f = return_f
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):

        if self.return_f:
            score = self.classifier(x)
            return score
        else:
            x = self.classifier(x)
            return x
    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck3_neck(nn.Module):
    def __init__(self, input_dim, feat_dim, return_f=False):
        super(BNNeck3_neck, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False, groups=feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)

        self.bn.bias.requires_grad_(False)
        self.bn.apply(self.weights_init_kaiming)


    def forward(self, x):
        x = self.reduction(x)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)

        return after_neck, before_neck

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck3_classifier(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3_classifier, self).__init__()
        self.return_f = return_f


        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):

        if self.return_f:
            score = self.classifier(x)
            return score
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

class BNNeck3_depthwise(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3_depthwise, self).__init__()
        self.return_f = return_f
        # 1x1 Convolution으로 차원 축소: (N, input_dim, H, W) -> (N, feat_dim, H, W)
        self.reduction = nn.Conv2d(input_dim, feat_dim, 1, bias=False, groups=512)
        # 4D 텐서를 받는 BatchNorm2d 사용 (BN2d는 (N, C, H, W)를 기대)
        self.bn = nn.BatchNorm2d(feat_dim)
        # BN의 bias는 학습하지 않음.
        self.bn.bias.requires_grad_(False)
        # 분류기: 최종 출력 피처는 feat_dim 차원 (flatten 후 적용)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        # x: (N, input_dim, H, W)
        x = self.reduction(x)  # (N, feat_dim, H, W)
        # 만약 spatial 차원이 1이 아니라면 pooling 적용하여 (N, feat_dim, 1, 1)로 변환
        if x.size(2) != 1 or x.size(3) != 1:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        # BatchNorm2d 적용 (입력은 4D: (N, feat_dim, 1, 1))
        bn_out = self.bn(x)
        # flatten하여 (N, feat_dim) 형태로 변환
        feat = bn_out.view(bn_out.size(0), -1)
        if self.return_f:
            score = self.classifier(feat)
            # 원본 (풀링 전) 텐서도 flatten해서 반환할 수 있음
            pre_bn = x.view(x.size(0), -1)
            return feat, score, pre_bn
        else:
            out = self.classifier(feat)
            return out

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

