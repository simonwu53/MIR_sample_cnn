from src.models import build_model, get_loss, load_ckpt, apply_state_dict, AUCMetric
from src.data import DataPrefetcher, MTTDataset
from src.utils import LOG, CONSOLE, MTT_MEAN, MTT_STD
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from pathlib import Path
import time


def test_on_model(args):
    device = args.device
    if device == 'cpu':
        raise NotImplementedError("CPU training is not implemented.")
    p_out = Path(args.p_out).joinpath(f"{args.m}^{args.n}-model-{args.tensorboard_exp_name}")
    if not p_out.exists():
        p_out.mkdir(exist_ok=True, parents=True)

    # build model
    model = build_model(args)
    model.to(device)

    # dataset & loader
    test_dataset = MTTDataset(path=args.p_data, split='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.n_workers,
                             pin_memory=True,
                             drop_last=False)                                    # not dropping last in testing
    test_steps = test_dataset.calc_steps(args.batch_size, drop_last=False)       # not dropping last in testing
    LOG.info(f"Total testing steps: {test_steps}")
    LOG.info(f"Testing data size: {len(test_dataset)}")

    # create loss
    loss_fn = get_loss(args.loss)
    # create metric
    metric = AUCMetric()

    # load checkpoint OR init state_dict
    if args.checkpoint is not None:
        state_dict = load_ckpt(args.checkpoint,
                               reset_epoch=args.ckpt_epoch,
                               no_scheduler=args.ckpt_no_scheduler,
                               no_optimizer=args.ckpt_no_optimizer,
                               no_loss_fn=args.ckpt_no_loss_fn,
                               map_values=args.ckpt_map_values)
        model_dict = {'model': model} if 'model' in state_dict else None
        apply_state_dict(state_dict, model=model_dict)
        best_val_loss = state_dict['val_loss']
        epoch = state_dict['epoch']
        global_i = state_dict['global_i']
        LOG.info(f"Checkpoint loaded. Epoch trained {epoch}, global_i {global_i}, best val {best_val_loss:.6f}")
    else:
        raise AssertionError("Pre-trained checkpoint must be provided.")

    # summary writer
    writer = SummaryWriter(log_dir=p_out.as_posix(), filename_suffix='-test')

    # start testing
    model.eval()
    model.test_mode(True)
    status_col = TextColumn("")
    running_loss = 0
    if args.data_normalization:
        fetcher = DataPrefetcher(test_loader, mean=MTT_MEAN, std=MTT_STD)
    else:
        fetcher = DataPrefetcher(test_loader, mean=None, std=None)
    samples, targets = fetcher.next()

    with Progress("[progress.description]{task.description}",
                  "[{task.completed}/{task.total}]",
                  BarColumn(),
                  "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeRemainingColumn(),
                  TextColumn("/"),
                  TimeElapsedColumn(),
                  status_col,
                  expand=False, console=CONSOLE, refresh_per_second=5) as progress:
        task = progress.add_task(description=f'[Test]', total=test_steps)
        i = 0  # counter
        t_start = time.time()

        with torch.no_grad():
            while samples is not None:
                # forward model
                out = model(samples)
                test_loss = loss_fn(out, targets)

                # collect running loss
                running_loss += test_loss.item()
                i += 1
                writer.add_scalar('Test/Loss', running_loss / i, i)

                # auc metric
                metric.step(targets.cpu().numpy(), out.cpu().numpy())

                # pre-fetch next samples
                samples, targets = fetcher.next()

                if not progress.finished:
                    status_col.text_format = f"Test loss: {running_loss/i:.06f}"
                    progress.update(task, advance=1)

    auc_tag, auc_sample, ap_tag, ap_sample = metric.auc_ap_score
    LOG.info(f"Testing speed: {(time.time() - t_start)/i:.4f}s/it, "
             f"auc_tag: {auc_tag:.04f}, "
             f"auc_sample: {auc_sample:.04f}, "
             f"ap_tag: {ap_tag:.04f}, "
             f"ap_sample: {ap_sample:.04f}")
    writer.close()
    return
