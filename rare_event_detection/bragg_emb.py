import logging
import os
import torch

from .dataset import BraggDataset
from time import perf_counter
from pathlib import Path
from torch.utils.data import DataLoader

from .model import BraggPeakBYOL, MLPhead, targetNN

def regression_loss(x, y):
    x = torch.nn.functional.normalize(x, dim=-1, p=2)
    y = torch.nn.functional.normalize(y, dim=-1, p=2)
    
    return 2 - 2 * (x * y).sum(dim=-1)

def train_bragg_embedding(training_scan_file: Path,
                         training_dark_file: Path,
                         itr_out_dir: Path, # iterated output directory for trained models
                         thold: int=100, # threshold value to process raw images
                         psz: int=15, # training model patch size
                         maxep: int=1000, # max training epochs
                         mbsz: int=256, # mini batch size
                         nworks: int=8, # number of workers for data loading
                         lr: float=1e-3, # learning rate
                         zdim: int=256): # projection dimension
    total_time = 0
    total_comp_time = 0
    total_2dev_time = 0
    total_cpul_time = 0
    total_logg_time = 0

    data_obtain_time1 = 0
    data_obtain_time2 = 0

    total_time_tick = perf_counter()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    os.environ["WORLD_SIZE"] = "1"

    torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("[Info] loading data into CPU memory, it will take a while ... ...")

    print(torch_devs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    time_datl_tick = perf_counter()

    # modified data loader here to read the raw file 
    train_ds = BraggDataset(training_scan_file, training_dark_file, thold=thold, psz=psz, train=True) 

    print(f'data load phase 1 time is {perf_counter()-time_datl_tick}')

    time_datl_tick = perf_counter()

    train_dl = DataLoader(dataset=train_ds, batch_size=mbsz, shuffle=True,\
                   num_workers=nworks, prefetch_factor=mbsz, drop_last=True, pin_memory=True)
    logging.info("%d samples loaded for training" % len(train_ds))

    print(f'data load phase 2 time is {perf_counter()-time_datl_tick}')
    
    time_mdli_tick = perf_counter()

    model = BraggPeakBYOL(psz=psz, hdim=64, proj_dim=zdim)
    predictor = MLPhead(zdim, zdim, zdim)

    print(f'model initial phase 1 time is {perf_counter()-time_mdli_tick}')

    time_mdli_tick = perf_counter()

    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        print(gpus)
        if gpus > 1:
            logging.info("This implementation only makes use of one GPU although %d are visiable" % gpus)
        model = model.to(torch_devs)
        predictor = predictor.to(torch_devs)
    
    print(f'model initial phase 2 time is {perf_counter()-time_mdli_tick}')

    time_mdli_tick = perf_counter()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr) 

    target_maker = targetNN(beta=0.996)
    target_maker.update(model)

    print(f'model initial phase 3 time is {perf_counter()-time_mdli_tick}')

    for epoch in range(maxep):
        ep_tick = perf_counter()
        time_comp = 0
        prev_iter_end_time = None

        for v1, v2 in train_dl:
            time_2dev_tick = perf_counter()

            if prev_iter_end_time is not None:
                total_cpul_time += time_2dev_tick - prev_iter_end_time
            else:
                total_cpul_time += time_2dev_tick - ep_tick

            start.record()
            v1 = v1.to(torch_devs)
            v2 = v2.to(torch_devs)
            
            end.record()
            torch.cuda.synchronize()
            total_2dev_time += start.elapsed_time(end)

            # print(start.elapsed_time(end))
            # total_2dev_time += start.elapsed_time(end)
            # total_2dev_time += perf_counter() - time_2dev_tick

            it_comp_tick = perf_counter()
            start.record()

            optimizer.zero_grad()

            y_rep_v1, z_proj_v1 = model.forward(v1, rety=False)
            y_rep_v2, z_proj_v2 = model.forward(v2, rety=False)

            online_p1 = predictor.forward(z_proj_v1)
            online_p2 = predictor.forward(z_proj_v2)

            target_p1 = target_maker.predict(v1)
            target_p2 = target_maker.predict(v2)
            
            loss = regression_loss(online_p1, target_p2) + regression_loss(online_p2, target_p1) #+\
                #    y_rep_v1.abs().sum(axis=-1) + y_rep_v2.abs().sum(axis=-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step() 

            target_maker.update(model)

            time_comp += 1000 * (perf_counter() - it_comp_tick)
            end.record()
            torch.cuda.synchronize()
            total_comp_time += start.elapsed_time(end)

            prev_iter_end_time = perf_counter()

        time_e2e = 1000 * (perf_counter() - ep_tick)
        _prints = '[Info] @ %.1f Epoch: %05d, loss: %.4f, elapse: %.2fms/epoch (computation=%.1fms/epoch, %.2f%%)' % (\
                   perf_counter(), epoch, loss.cpu().detach().numpy(), time_e2e, time_comp, 100*time_comp/time_e2e)
        logging.info(_prints)

        start.record()
        torch.jit.save(torch.jit.trace(model, v1), "%s/script-ep%05d.pth" % (itr_out_dir, epoch+1))
        end.record()
        torch.cuda.synchronize()
        total_logg_time += start.elapsed_time(end)

    total_time = perf_counter() - total_time_tick
    print(f"total 2 dev time is {total_2dev_time/1000.0}, \ntotal real compute time is {total_comp_time/1000.0}, \
    \ntotal model saving time is {total_logg_time/1000.0}, \ntotal loading from cpu time is {total_cpul_time}, \
    \nthe total time is {total_time} s")
