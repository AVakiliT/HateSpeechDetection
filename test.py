import datetime
import random

from torch.utils import tensorboard
from torch.utils.tensorboard.summary import hparams



for batch in [32, 64]:
    for lr in [.001, .0002]:
        time_string = (datetime.datetime.utcnow() + datetime.timedelta(seconds=12600)).replace(
            microsecond=0).isoformat('_')

        tb = tensorboard.SummaryWriter(log_dir=f'runs/batch={batch}--lr={lr}/{time_string}')

        for i in range(20):
            acc = random.random()
            f1 = random.random()
            tb.add_scalar('train_acc', acc, i)
            tb.add_scalar('train_f1', f1, i)
            tb.add_scalar('test_acc', acc, i)
            tb.add_scalar('test_f1', f1, i)
            # tb.add_scalars('train', {'acc': acc, 'f1': f1}, i)
            # tb.add_scalars('test', {'acc': acc, 'f1': f1}, i)


        a = ({
                 'batch': batch,
                 'lr': lr
             },
             {
                 'zfinal_acc': acc,
                 'zfinal_f1': f1
             })

        for j in hparams(*a):
            print(batch, lr)
            tb.file_writer.add_summary(j)

        for k, v in a[1].items():
            tb.add_scalar(k, v)

        # tb.add_hparams(*a)


