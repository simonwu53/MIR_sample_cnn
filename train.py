from src.models import build_model, get_loss, get_optimizer
from src.data import DataPrefetcher, MTTDataset
from src.utils import VAR, LOG, CONSOLE, MTT_MEAN, MTT_STD
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn


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
                    status_col.text_format = f"Val loss: {running_loss/i:.04f}"
                    progress.update(task, advance=1)
    return running_loss / i


def train_one_epoch(model, optim, loss_fn, loader, epoch, steps):
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
            # pre-fetch next samples
            samples, targets = fetcher.next()

            if not progress.finished:
                status_col.text_format = f"Loss: {running_loss/i:.04f}"
                progress.update(task, advance=1)

    return running_loss / i


def train_epochs(model, init_epoch, args):
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

    # creating scheduler TODO Tensorboard CSVLogger
    lr_sche_plateau = ReduceLROnPlateau(optim,
                                        factor=args.lr_decay_plateau,
                                        patience=args.plateau_patience,
                                        min_lr=args.min_lr, verbose=True)

    # train on epochs
    for i in range(init_epoch, args.max_epoch):
        # train
        train_loss = train_one_epoch(model, optim, loss_fn, train_loader, i+1, train_steps)

        # validate
        val_loss = evaluate(model, loss_fn, val_loader, i+1, val_steps)

        # update scheduler
        lr_sche_plateau.step(val_loss)
    return


def train_on_model(args):
    device = args.device
    if device == 'cpu':
        raise NotImplementedError("CPU training is not implemented.")

    # build model
    model = build_model(args)
    model.to(device)

    # TODO load checkpoint
    current_train = 0

    # TODO resume 'retrain'
    for i in range(current_train, args.max_train):
        CONSOLE.rule(f"Start stage {i}")
        # TODO find best model
        init_epoch = 0

        # calculate initial lr
        decay = args.lr_decay ** i
        args.lr = args.init_lr * decay

        # train on epochs
        train_epochs(model, init_epoch, args)
    return
