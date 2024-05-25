import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss

# CORE: COnsistent REpresentation Learning for Face Forgery Detection, CVPRW 22
# Revise by Sun
class ConsistencyCos(nn.Module):
    def __init__(self, device='cpu'):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()
        self.device = device

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        loss = self.mse_fn(cos, labels.to(self.device))
        return loss

class PolyBCELoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (∗), where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        self.bce_loss = self.bce(input, target)
        pt = torch.sigmoid(input) 
        pt = torch.where(target==1,pt,1-pt)
        poly_loss = self.bce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)
    


