import os
import torch
from tqdm import tqdm


def get_train_configs(config):
    return config["MODEL_PATH"], config["SAVE"], config["EPOCHS"]


def correct(outputs, Y):
    pred = torch.argmax(outputs, dim = 1)
    return torch.sum(pred == Y)


def train(net, dataloader, criterion, optimizer, device, config, version):
    model_path, save, epochs = get_train_configs(config)
    min_loss = 1e9
    for e in range(1, epochs + 1):
        running_loss = 0
        count = 0
        runner = tqdm(enumerate(dataloader), total = len(dataloader))
        for dex, batch in runner:
            x, y = batch
            x, y = x.to(device), y.to(device)
            batch_size = x.size()[0]
            h, c = net.zero_state(batch_size)
            h = h.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            o = net(x, (h, c))
            loss = criterion(o, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
            epoch_loss = running_loss / count
            runner.set_postfix(avg_loss = '{:4f}'.format(epoch_loss))
        print('Epoch: %d/%d | Loss: %f' % (e, epochs, epoch_loss))
        
        if save == True and running_loss < min_loss:
            min_loss = running_loss
            save_path = os.path.join(model_path, '{}_e{}.pth'.format(version, e))
            torch.save(net.state_dict(), save_path)

    if save == True:
        save_path = os.path.join(model_path, '{}.pth'.format(version))
        torch.save(net.state_dict(), save_path)