import math
from torch.optim.lr_scheduler import LambdaLR


def get_inverse_square_root_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with an inverse square root decay preceded by a linear warmup.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_warmup_steps) / math.sqrt(float(max(1, current_step))))

    return LambdaLR(optimizer, lr_lambda, last_epoch) 
 