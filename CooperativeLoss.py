import torch
import torch.nn as nn

class CooperativeLoss(nn.Module):
    def __init__(self, lambda_coop=1.0, lambda_pena=1.0):
        """
        lambda_coop: weight for the cooperative loss component
        lambda_pena: weight for the aggression penalty component
        """

        super(CooperativeLoss, self).__init__()
        self.lambda_coop = lambda_coop
        self.lambda_pena = lambda_pena

    def forward(self, predicted, target, coop, penalty):
        """
        predicted: Predicted Q-values from the model (tensor)
        target: target Q-values calculated from Bellman equation (tensor)
        coop: cooperative reward (float)

        penalty: Penalties incurred for aggressive actions (tensor or float).
        return: Computed loss value.
        """

        # the standard MSE loss between predicted and target Q-values
        mse_loss = torch.nn.functional.mse_loss(predicted, target)

        # cooperative loss component
        coop = torch.as_tensor(coop, device=predicted.device, dtype=predicted.dtype)
        if coop.dim() > 0:
            coop_loss = coop.mean()

        # penalty loss component
        penalty = torch.as_tensor(penalty, device=predicted.device, dtype=predicted.dtype)
        if penalty.dim() > 0:
            penalty_loss = penalty.mean()

        # total loss
        total_loss = mse_loss - self.lambda_coop * coop_loss + self.lambda_pena * penalty_loss
        return total_loss
