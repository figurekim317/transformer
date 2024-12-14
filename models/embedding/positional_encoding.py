import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding
    """

    def __init__(self, d_model, max_len, device):
        """
        Constructor of sinusoid encoding class
        param d_model: dimensional of model
        param max_len: max sequence length
        param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        # 부모클래스의 메서드를 호출하는데 사용함
        # 클래스 상속을 사용할 때, 부모 클래스의 초기화 작업이나 메서드를 자식 클래스에서 호출하려면
        # pytorch에서 사용자 정의 클래스를 만들 때 nn.Module을 상속받으면, 
        # nn.Module의 초기화 과정이 포함된 __init__ 메서드를 호출해야 한다.
        # 이 초기화 과정을 생략하면 pytorch 내부에서 사용하는 중요한 기능이 제대로 동작하지 않음

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) 
        # unsqueeze이므로 차원을 하나 추가함 (max_len, 1) 형태로 만듦
        # broadcasting을 통해 계산을 쉽게 하기 위함
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]
    