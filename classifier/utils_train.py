import torch
import time

def train_one_epoch(model, train_loader, epoch_index, tb_writer, device, optimizer, loss_fn):
    all_train_labels = []
    all_train_preds = []

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        start_time = time.time()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.cpu().item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        # predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        # accumulate labels/preds
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(preds.cpu().numpy())

        elapsed_iter_time = time.time() - start_time
        eta = elapsed_iter_time * (len(train_loader) - i)
        print(f"[{i+1} / {len(train_loader)}] iter: {elapsed_iter_time:.2f} sec | eta: {eta:.2f} sec")

    return all_train_labels, all_train_preds, last_loss


def get_split_filename(k_type, val_size, test_size, random_state):
    return f"dataset_{k_type}_vs{val_size}_ts{test_size}_rs{random_state}_split_indices.csv"
