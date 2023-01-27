## Scheduler
This section implements some special scheduler for model training.

## Implementations
### [LinearWarmupCosineAnnealingLR.py](https://github.com/come880412/DL_common/blob/main/scripts/scheduler/LinearWarmupCosineAnnealingLR.py)
"LinearWarmupCosineAnnealing" Learning rate scheduler trick decaying by epoch.
```bash
python scripts/scheduler/LinearWarmupCosineAnnealingLR.py <warmup_epochs> <max_epochs> <warmup_start_lr> <eta_min>
```
- warmup_epochs: Maximum number of epochs for linear warmup. Default:10
- max_epochs: Maximum number of epochs. Default:100
- warmup_start_lr: Learning rate to start the linear warmup. Default: 0.
- eta_min: Minimum learning rate. Default: 0.

<p align="center">
<img src="https://github.com/come880412/DL_common/blob/main/images/lr_decay.png" width=60% height=60%>
</p>