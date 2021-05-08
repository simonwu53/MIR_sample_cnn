from src.models import build_model, get_loss, get_optimizer, load_ckpt, find_optimal_model, apply_lr, EarlyStopping, ReduceLROnPlateau
from src.data import DataPrefetcher, MTTDataset
from src.utils import VAR, LOG, CONSOLE, MTT_MEAN, MTT_STD
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from pathlib import Path
import time


def evaluate(model, loss_fn, loader, epoch, steps, normalize=None):
    model.eval()
    status_col = TextColumn("")
    running_loss = 0

    if normalize is not None:
        assert len(normalize) == 2, "mean and std values should be provided to use data normalization"
        fetcher = DataPrefetcher(loader, mean=normalize[0], std=normalize[1])  # modified behavior - w/ input normalization
    else:
        fetcher = DataPrefetcher(loader, mean=None, std=None)                  # original behavior - no input normalization
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
        task = progress.add_task(description=f'[Eval  {epoch}]', total=steps)
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
                    status_col.text_format = f"Val loss: {running_loss/i:.06f} " \
                                             f"speed: {(time.time() - t_start)/i:.4f}s/it"
                    progress.update(task, advance=1)
    return running_loss / i


def train_one_epoch(model, optim, loss_fn, loader, epoch, steps, writer, global_i, writer_interval=200, normalize=None):
    model.train()
    status_col = TextColumn("")
    running_loss = 0
    lr = optim.param_groups[0]['lr']

    if normalize is not None:
        assert len(normalize) == 2, "mean and std values should be provided to use data normalization"
        fetcher = DataPrefetcher(loader, mean=normalize[0], std=normalize[1])  # modified behavior - w/ input normalization
    else:
        fetcher = DataPrefetcher(loader, mean=None, std=None)                  # original behavior - no input normalization
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
        task = progress.add_task(description=f'[Epoch {epoch}]', total=steps)
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
                status_col.text_format = f"Loss: {running_loss/i:.06f} " \
                                         f"speed: {(time.time() - t_start)/i:.4f}s/it " \
                                         f"lr: {lr}"
                progress.update(task, advance=1)

    return running_loss / i, global_i


def apply_state_dict(state_dict, model=None, optim=None, loss_fn=None, scheduler=None):
    if model is not None:
        k, v = list(model.items())[0]
        v.load_state_dict(state_dict[k])
    if optim is not None:
        k, v = list(optim.items())[0]
        v.load_state_dict(state_dict[k])
    if loss_fn is not None:
        k, v = list(loss_fn.items())[0]
        v.load_state_dict(state_dict[k])
    if scheduler is not None:
        k, v = list(scheduler.items())[0]
        v.load_state_dict(state_dict[k])
    return


def train_on_model(args):
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
    if args.data_normalization:
        normalize = (MTT_MEAN, MTT_STD)
        LOG.info("Data normalization [bold cyan]on[/]")
    else:
        normalize = None
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
                                          min_lr=args.min_lr,
                                          verbose=True,
                                          prefix="[Scheduler Plateau] ",
                                          logger=LOG)
    scheduler_es = EarlyStopping(patience=args.early_stop_patience,
                                 min_delta=args.early_stop_delta,
                                 verbose=True,
                                 prefix="[Scheduler Early Stop] ",
                                 logger=LOG)

    # load checkpoint OR init state_dict
    if args.checkpoint is not None:
        state_dict = load_ckpt(args.checkpoint,
                               reset_epoch=args.ckpt_epoch,
                               no_scheduler=args.ckpt_no_scheduler,
                               no_optimizer=args.ckpt_no_optimizer,
                               no_loss_fn=args.ckpt_no_loss_fn,
                               map_values=args.ckpt_map_values)
        model_dict = {'model': model} if 'model' in state_dict else None
        optim_dict = {'optim': optim} if 'optim' in state_dict else None
        loss_fn_dict = {'loss_fn': loss_fn} if 'loss_fn' in state_dict else None
        scheduler_dict = {'scheduler_plateau': scheduler_plateau} \
            if 'scheduler_plateau' in state_dict else None
        apply_state_dict(state_dict,
                         model=model_dict,
                         optim=optim_dict,
                         loss_fn=loss_fn_dict,
                         scheduler=scheduler_dict)
        best_val_loss = state_dict['val_loss']
        epoch = state_dict['epoch']
        global_i = state_dict['global_i']
        LOG.info(f"Checkpoint loaded. Epoch trained {epoch}, global_i {global_i}, best val {best_val_loss:.6f}")
    else:
        # fresh training
        best_val_loss = 9999
        epoch = 0
        global_i = 0

    # tensorboard
    purge_step = None if global_i == 0 else global_i
    writer = SummaryWriter(log_dir=VAR
                           .log
                           .joinpath('tensorboard')
                           .joinpath(f"{args.m}^{args.n}-model-{args.tensorboard_exp_name}")
                           .as_posix(),
                           purge_step=purge_step,
                           filename_suffix='-train')

    # train model for epochs
    assert epoch < args.max_epoch, "Initial epoch value must be smaller than max_epoch in order to train model"
    for i in range(epoch, args.max_epoch):

        # train
        init_lr = optim.param_groups[0]['lr']
        train_loss, global_i = train_one_epoch(model, optim, loss_fn, train_loader,
                                               epoch+1, train_steps, writer, global_i,
                                               writer_interval=args.tensorboard_interval,
                                               normalize=normalize)

        # validate
        val_loss = evaluate(model, loss_fn, val_loader, epoch+1, val_steps, normalize=normalize)
        writer.add_scalar('Loss/Val', val_loss, global_i)

        epoch += 1

        # update scheduler
        scheduler_plateau.step(val_loss)
        scheduler_es.step(val_loss)

        # save checkpoint
        if optim.param_groups[0]['lr'] != init_lr:
            LOG.info(f"Saving [red bold]checkpoint[/] at epoch {epoch}, model saved to {p_out.as_posix()}")
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'scheduler_plateau': scheduler_plateau.state_dict(),
                'scheduler_es': scheduler_es.state_dict(),
                'epoch': epoch,
                'loss': train_loss,
                'val_loss': val_loss,
                'p_out': p_out,
                'global_i': global_i
            },
                p_out.joinpath(f'ckpt@epoch-{epoch:03d}-loss-{val_loss:.6f}.tar').as_posix())

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            LOG.info(f"New [red bold]best[/] validation loss {val_loss:.6f}, model saved to {p_out.as_posix()}")
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'scheduler_plateau': scheduler_plateau.state_dict(),
                'scheduler_es': scheduler_es.state_dict(),
                'epoch': epoch,
                'loss': train_loss,
                'val_loss': val_loss,
                'p_out': p_out,
                'global_i': global_i
            },
                p_out.joinpath(f'best@epoch-{epoch:03d}-loss-{val_loss:.6f}.tar').as_posix())

        # save latest model
        else:
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'scheduler_plateau': scheduler_plateau.state_dict(),
                'scheduler_es': scheduler_es.state_dict(),
                'epoch': epoch,
                'loss': train_loss,
                'val_loss': val_loss,
                'p_out': p_out,
                'global_i': global_i
            },
                p_out.joinpath(f'latest.tar').as_posix())

        # early stop, if enabled
        if scheduler_es.early_stop:
            break

        # if load optimal model when lr changed
        if optim.param_groups[0]['lr'] != init_lr and args.load_optimal_on_plateau:
            # save lr before restoring
            cur_lr = [param_group['lr'] for param_group in optim.param_groups]
            # restore last best model
            state_dict = find_optimal_model(p_out)
            apply_state_dict(state_dict,
                             model={'model': model},
                             optim={'optim': optim},
                             loss_fn=None,
                             scheduler=None)
            apply_lr(optim, cur_lr)
            # reset global_i
            global_i = state_dict['global_i']
            epoch = state_dict['epoch']
            LOG.info(f"Best model (val loss {state_dict['val_loss']}) applied. Roll back to epoch {epoch}")
            # reset tensorboard writer
            writer.close()
            writer = SummaryWriter(log_dir=VAR
                                   .log
                                   .joinpath('tensorboard')
                                   .joinpath(f"{args.m}^{args.n}-model-{args.tensorboard_exp_name}")
                                   .as_posix(),
                                   purge_step=global_i,
                                   filename_suffix='-train')

    # close tensorboard
    writer.close()
    return
