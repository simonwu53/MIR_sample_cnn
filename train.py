from src.models import build_model, get_loss, get_optimizer
from src.data import DataPrefetcher, MTTDataset
from src.utils import PATH, LOG, CONSOLE
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import track, Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn


def train_one_epoch(model, optim, loss_fn, loader, epoch, steps):
    model.train()
    status_col = TextColumn("")

    # TODO add mean std
    fetcher = DataPrefetcher(loader, mean=None, std=None)
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

        while samples is not None:
            if samples is None:
                LOG.warning("No data loaded.")
                break
            # zero the parameter gradients
            optim.zero_grad()
            # forward + backward + optimize
            out = model(samples)
            loss = loss_fn(out, targets)
            loss.backward()
            optim.step()
            # pre-fetch next samples
            samples, targets = fetcher.next()

            if not progress.finished:
                status_col.text_format = f"Loss: {loss.item():.04f}"
                progress.update(task, advance=1)

    return


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
    steps = train_dataset.calc_steps(args.batch_size)
    LOG.info(f"Total training steps: {steps}")
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
        train_one_epoch(model, optim, loss_fn, train_loader, i+1, steps)
        # validate
        # update scheduler
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
