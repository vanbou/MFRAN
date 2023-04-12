from model import common
import torch
import torch.nn as nn

def make_model(args):
    return MFRAN(args)


class MFRAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        n_feats = args.n_feats
        n_blocks = args.n_resblocks
        kernel_size = 3
        scale = args.scale[0]
        self.n_blocks = n_blocks
        self.div = 2

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        act = nn.ReLU(True)
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(common.ResFractalBlock(args.n_columns, n_feats, n_feats))

        # MSFFRN tail module
        modules_tail = [
            nn.Conv2d(n_feats, n_feats, kernel_size,padding=(kernel_size-1)//2, stride=1),
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats,  args.n_colors, kernel_size,padding=(kernel_size-1)//2, stride=1)
        ]

        self.adjust = nn.Sequential(nn.Conv2d((self.n_blocks//self.div +1) * n_feats, n_feats, 1, padding=0,  stride=1))
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.ca = ChannelAttention(n_feats)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        # self.skip7 = nn.Sequential(*skip7)

    def forward(self, x):

        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        MSFFRB_out = []
        for i in range(self.n_blocks // 2):
            x = self.body[i](x)
            MSFFRB_out.append(x)
        MSFFRB_out.append(res)
        res2 = x
        x = torch.cat(MSFFRB_out, 1)
        x = self.adjust(x)
        # mid = x * self.ca(x)
        mid = x
        x = mid

        MSFFRB_out2 = []
        for i in range(self.n_blocks // 2):
            x = self.body[i](x)
            MSFFRB_out2.append(x)
        MSFFRB_out2.append(res2)
        res = torch.cat(MSFFRB_out2, 1)
        x = self.adjust(res)
        mid2 = x
        mid2 += mid
        x = self.tail(mid2)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'.format(
                                name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(
                        'unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(
                    'missing keys in state_dict: "{}"'.format(missing))
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes //16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)