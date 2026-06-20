import torch
import torch.nn.functional as F


def train(args, model, train_loader, client_id=None, global_state=None):
    model.train()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

    first_batch_loss = None
    last_batch_loss = None
    loss_sum = 0.0
    sample_count = 0
    global_params = None
    if args.algorithm == "fedprox":
        if global_state is None:
            raise RuntimeError("FedProx requires the current global model state")
        global_params = {
            name: value.detach().to(args.device)
            for name, value in global_state.items()
        }
    for _ in range(args.E):
        for data, target in train_loader:
            data = data.to(args.device, non_blocking=args.pin_memory)
            target = target.to(args.device, non_blocking=args.pin_memory)

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = F.cross_entropy(logits, target.long())
            if global_params is not None and args.mu > 0:
                proximal_term = torch.zeros((), device=args.device)
                for name, parameter in model.named_parameters():
                    proximal_term = proximal_term + (
                        parameter - global_params[name].to(parameter.dtype)
                    ).square().sum()
                loss = loss + 0.5 * args.mu * proximal_term
            loss.backward()
            optimizer.step()

            if first_batch_loss is None:
                first_batch_loss = loss.item()
            last_batch_loss = loss.item()
            loss_sum += loss.item() * target.size(0)
            sample_count += target.size(0)

        if scheduler is not None:
            scheduler.step()

    if sample_count == 0:
        raise RuntimeError(f"Client {client_id} has no training samples")

    return {
        "final_loss": float(last_batch_loss),
        "mean_loss": loss_sum / sample_count,
        "first_batch_loss": float(first_batch_loss),
        "last_batch_loss": float(last_batch_loss),
        "loss_drop": float(first_batch_loss - last_batch_loss),
    }


def evaluate_batch(model, data, target):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = F.cross_entropy(logits, target.long()).item()
        accuracy = logits.argmax(dim=1).eq(target).float().mean().item()
    if was_training:
        model.train()
    return loss, accuracy


def test(args, model, data_loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(args.device, non_blocking=args.pin_memory)
            target = target.to(args.device, non_blocking=args.pin_memory)
            logits = model(data)
            loss_sum += F.cross_entropy(
                logits,
                target.long(),
                reduction="sum",
            ).item()
            correct += logits.argmax(dim=1).eq(target).sum().item()
            sample_count += target.size(0)

    return loss_sum / sample_count, correct / sample_count
