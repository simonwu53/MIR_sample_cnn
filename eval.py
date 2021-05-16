import numpy as np
import pandas as pd
from src.models import build_model, get_loss, load_ckpt, apply_state_dict, AUCMetric
from src.data import DataPrefetcher, MTTDataset, _load_audio, _segment_audio
from src.utils import LOG, CONSOLE, MTT_MEAN, MTT_STD
import torch
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from pathlib import Path
import time


def test_on_model(args):
    device = args.device
    if device == 'cpu':
        raise NotImplementedError("CPU training is not implemented.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # build model
    model = build_model(args)
    model.to(device)

    # output dir
    p_out = Path(args.p_out).joinpath(f"{model.name}-{args.tensorboard_exp_name}")
    if not p_out.exists():
        p_out.mkdir(exist_ok=True, parents=True)

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
    sigmoid = Sigmoid().to(device)
    status_col = TextColumn("")
    running_loss = 0
    if args.data_normalization:
        fetcher = DataPrefetcher(test_loader, mean=MTT_MEAN, std=MTT_STD, device=device)
    else:
        fetcher = DataPrefetcher(test_loader, mean=None, std=None, device=device)
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
                logits = model(samples)
                out = sigmoid(logits)
                test_loss = loss_fn(logits, targets)

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


def eval_on_model(args):
    device = args.device
    if device == 'cpu':
        raise NotImplementedError("CPU training is not implemented.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # build model
    model = build_model(args)
    model.to(device)

    # output dir
    p_out = Path(args.p_out).joinpath(f"{model.name}-{args.tensorboard_exp_name}")
    if not p_out.exists():
        p_out.mkdir(exist_ok=True, parents=True)

    # dataset & loader
    annotation = pd.read_csv(args.annotation_file)
    query = annotation[annotation.mp3_path.str.match('/'.join(args.audio_file.split('/')[-2:]))]
    assert query.shape[0] != 0, f"Cannot find the audio file: {args.audio_file}"
    # split audio info and segment audio
    threshold = args.eval_threshold
    song_info = query[query.columns.values[50:]]
    tags = query.columns.values[:50]
    labels = query[tags].values[0]
    label_names = tags[labels.astype(bool)]
    segments = _segment_audio(_load_audio(args.audio_file, sample_rate=22050), n_samples=59049)
    LOG.info(f"Song info: {song_info}")
    LOG.info(f"Number of segments: {len(segments)}")
    LOG.info(f"Ground truth tags: {label_names}")
    LOG.info(f"Positive tag threshold: {threshold}")

    # create loss
    loss_fn = get_loss(args.loss)

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

    # start testing
    model.eval()
    sigmoid = Sigmoid().to(device)
    t_start = time.time()

    # concatenate segments
    segments = torch.from_numpy(np.concatenate([seg.reshape(1, 1, -1) for seg in segments])).to(torch.float32).cuda(device=device)
    targets = torch.from_numpy(np.concatenate([labels.reshape(1, -1)]*10)).to(torch.float32).cuda(device=device)

    # forward pass
    with torch.no_grad():
        logits = model(segments)
        out = sigmoid(logits)
        loss = loss_fn(logits, targets)

    out = out.cpu().numpy()
    out[out > threshold] = 1
    out[out <= threshold] = 0
    out = np.sum(out, axis=0)
    res = pd.DataFrame(data={'tags': tags, 'freq': out})
    res = res[res.freq != 0].sort_values(by='freq', ascending=False)
    CONSOLE.print(res)

    LOG.info(f"Testing speed: {time.time() - t_start:.4f}s, "
             f"loss: {loss.item()}, ")
    return
