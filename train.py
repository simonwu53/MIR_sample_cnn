from src.models import build_model, get_loss, get_optimizer, find_optimal_model, EarlyStopping
from src.data import DataPrefetcher, MTTDataset
from src.utils import VAR, LOG, CONSOLE, MTT_MEAN, MTT_STD
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from pathlib import Path
import time


def evaluate(model, loss_fn, loader, epoch, steps):
    model.eval()
    status_col = TextColumn("")
    running_loss = 0

    fetcher = DataPrefetcher(loader, mean=MTT_MEAN, std=MTT_STD)
    samples, targets = fetcher.next()

    with Progress("[progress.description]{task.description}",
                  BarColumn(),
                  "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeRemainingColumn(),
                  TextColumn("/"),
                  TimeElapsedColumn(),
                  "{task.completed} of {task.total} steps",
                  status_col,
                  expand=False, console=CONSOLE, refresh_per_second=5) as progress:
        task = progress.add_task(description=f'[Eval  {epoch}] ', total=steps)
        i = 0  # counter
        t_start = time.time()

        with torch.no_grad():
            while samples is not None:

                # forward only
                out = model(samples)
                val_loss = loss_fn(out, targets)

                # collect running loss
                running_loss += val_loss.item()
                i += 1
                # pre-fetch next samples
                samples, targets = fetcher.next()

                if not progress.finished:
                    status_col.text_format = f"Val loss: {running_loss/i:.06f}"
                    progress.update(task, advance=1)
    return running_loss / i, (time.time() - t_start) / i


def train_one_epoch(model, optim, loss_fn, loader, epoch, steps, writer, global_i, writer_interval=200):
    model.train()
    status_col = TextColumn("")
    running_loss = 0

    fetcher = DataPrefetcher(loader, mean=MTT_MEAN, std=MTT_STD)
    samples, targets = fetcher.next()

    with Progress("[progress.description]{task.description}",
                  BarColumn(),
                  "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeRemainingColumn(),
                  TextColumn("/"),
                  TimeElapsedColumn(),
                  "{task.completed} of {task.total} steps",
                  status_col,
                  expand=False, console=CONSOLE, refresh_per_second=5) as progress:
        task = progress.add_task(description=f'[Epoch {epoch}] ', total=steps)
        i = 0  # counter
        has_graph = False
        t_start = time.time()

        while samples is not None:
            # zero the parameter gradients
            optim.zero_grad()
            # forward + backward + optimize
            out = model(samples)
            loss = loss_fn(out, targets)
            loss.backward()
            optim.step()

            # collect running loss
            running_loss += loss.item()
            i += 1
            global_i += 1

            # update tensorboard
            if i % writer_interval == 0:
                writer.add_scalar('Loss/Train', running_loss/i, global_i)
            if not has_graph:
                writer.add_graph(model, samples)
                has_graph = True

            # pre-fetch next samples
            samples, targets = fetcher.next()

            # update trackbar
            if not progress.finished:
                status_col.text_format = f"Loss: {running_loss/i:.06f}"
                progress.update(task, advance=1)

    return running_loss / i, global_i, (time.time() - t_start) / i


def train_epochs(model, state_dict, stage, args):
    # dataset & loader
    train_dataset = MTTDataset(path=args.p_data, split='train')
    val_dataset = MTTDataset(path=args.p_data, split='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.n_workers,
                            pin_memory=True,
                            drop_last=True)
    train_steps = train_dataset.calc_steps(args.batch_size)
    val_steps = val_dataset.calc_steps(args.batch_size)
    LOG.info(f"Total training steps: {train_steps}")
    LOG.info(f"Total validation steps: {val_steps}")
    LOG.info(f"Training data size: {len(train_dataset)}")
    LOG.info(f"Validation data size: {len(val_dataset)}")

    # create optimizer
    optim = get_optimizer(model.parameters(), args=args)

    # create loss
    loss_fn = get_loss(args.loss)

    # creating scheduler
    scheduler_plateau = ReduceLROnPlateau(optim,
                                          factor=args.lr_decay_plateau,
                                          patience=args.plateau_patience,
                                          min_lr=args.min_lr, verbose=True)
    scheduler_lambda = LambdaLR(optim,
                                lr_lambda=lambda epoch: args.lr_decay_local)
    if args.early_stop_patience is not None:
        assert args.early_stop_patience > 0, "Please specify the correct early stop patience value."
        scheduler_es = EarlyStopping(patience=args.early_stop_patience,
                                     verbose=True,
                                     prefix="[Scheduler]",
                                     logger=LOG)
    else:
        scheduler_es = None

    # load state_dict
    if 'model' in state_dict.keys():
        model.load_state_dict(state_dict['model'])
        optim.load_state_dict(state_dict['optim'])
        loss_fn.load_state_dict(state_dict['loss_fn'])
        scheduler_plateau.load_state_dict(state_dict['scheduler_plateau'])
        scheduler_lambda.load_state_dict(state_dict['scheduler_lambda'])
    best_val_loss = state_dict['val_loss']
    final_train_loss = state_dict['loss']
    init_epoch = state_dict['epoch']
    p_out = state_dict['p_out']

    # tensorboard
    global_i = state_dict['global_i']
    purge_step = None if global_i == 0 else global_i
    writer = SummaryWriter(log_dir=VAR.log.joinpath('tensorboard').joinpath(f"{args.m}^{args.n}-model").as_posix(),
                           purge_step=purge_step,
                           filename_suffix='-train')

    # train on epochs
    for i in range(init_epoch, args.max_epoch):
        # train
        train_loss, global_i, t_train = train_one_epoch(model, optim, loss_fn, train_loader,
                                                        i+1, train_steps, writer, global_i,
                                                        writer_interval=args.tensorboard_interval)

        # validate
        val_loss, t_val = evaluate(model, loss_fn, val_loader, i+1, val_steps)
        writer.add_scalar('Loss/Val', val_loss, global_i)

        LOG.info(f"Training time: {t_train:.4f}s per batch, Validation time {t_val:.4f}s per batch")

        # save the best model TODO add early stop state dict
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_train_loss = train_loss
            LOG.info(f"New best validation loss {val_loss:.6f}, model saved to {p_out.as_posix()}")
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'scheduler_plateau': scheduler_plateau.state_dict(),
                'scheduler_lambda': scheduler_lambda.state_dict(),
                'stage': stage,
                'epoch': i,
                'loss': train_loss,
                'val_loss': val_loss,
                'p_out': p_out,
                'global_i': global_i
            },
                p_out.joinpath(f'stage-{stage:02d}-epoch-{i:03d}-loss-{val_loss:.6f}.tar').as_posix())

        # update scheduler
        scheduler_plateau.step(val_loss)
        scheduler_lambda.step()
        if scheduler_es is not None:
            scheduler_es.step(val_loss)
            if scheduler_es.early_stop:
                break  # early stop

    # close tensorboard
    writer.close()
    return final_train_loss, best_val_loss


def train_on_model(args):
    device = args.device
    if device == 'cpu':
        raise NotImplementedError("CPU training is not implemented.")
    p_out = Path(args.p_out).joinpath(f"{args.m}^{args.n}-model")
    if not p_out.exists():
        p_out.mkdir(exist_ok=True, parents=True)

    # build model
    model = build_model(args)
    model.to(device)

    # load checkpoint & init state_dict
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
    else:
        state_dict = {
            'p_out': p_out,
            'stage': 0,
            'val_loss': 9999,
            'loss': 9999,
            'epoch': 0,
            'global_i': 0
        }
    current_train = state_dict['stage']

    # train model for N stages
    assert current_train < args.max_train, "Training stage is larger than max number of stages. Model is trained."
    for i in range(current_train, args.max_train):
        CONSOLE.rule(f"Start stage {i}")

        # calculate initial lr
        decay = args.lr_decay_global ** i
        args.lr = args.init_lr * decay

        # train on epochs
        train_loss, val_loss = train_epochs(model, state_dict, i, args)

        # find best model
        LOG.info(f'Stage {i} completed. Load overall optimal model for next stage')
        if i != args.max_train - 1:
            state_dict = find_optimal_model(p_out)
            LOG.info(f"Model (stage {state_dict['stage']} epoch {state_dict['epoch']} val {state_dict['val_loss']}) was selected for next stage")
    return
