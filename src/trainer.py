import os
import torch
from tqdm import tqdm


def get_train_configs(config):
    return config["MODEL_PATH"], config["SAVE"], config["EPOCHS"]


def correct(outputs, Y):
    pred = torch.argmax(outputs, dim = 1)
    return pred.eq(Y).sum().item()


def train(net, dataloader, criterion, optimizer, device, config, version):
    model_path, save, epochs = get_train_configs(config)
    max_acc = 0
    for e in range(1, epochs + 1):
        for mode in ["TRAIN", "VAL"]:
            running_loss = 0
            running_correct = 0
            count = 0
            runner = tqdm(enumerate(dataloader[mode]), total = len(dataloader[mode]), desc = mode)
            
            if mode == "TRAIN":
                net.train()
                for dex, batch in runner:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    batch_size = x.size()[0]
                    optimizer.zero_grad()
                    o = net(x)
                    loss = criterion(o, y)
                    loss.backward()
                    optimizer.step()
                    running_correct += correct(o, y)
                    running_loss += loss.item()
                    count += y.size()[0]
                    avg_loss = running_loss / count
                    accuracy = running_correct / count
                    runner.set_postfix(accuracy = '{:4f}'.format(100 * accuracy), avg_loss = '{:4f}'.format(avg_loss))
           
                print('{} \t  Epoch: {}/{} \t Loss: {:4f} \t Accuracy {:4f}'.format(mode, e, epochs, running_loss, accuracy))
            
            elif mode == "VAL":
                net.eval()
                for dex, batch in runner:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    batch_size = x.size()[0]
                    o = net(x)
                    loss = criterion(o, y)
                    running_correct += correct(o, y)
                    running_loss += loss.item()
                    count += y.size()[0]
                    avg_loss = running_loss / count
                    accuracy = running_correct / count
                    runner.set_postfix(accuracy = '{:4f}'.format(100 * accuracy), avg_loss = '{:4f}'.format(avg_loss))

                print('{} \t  Epoch: {}/{} \t Loss: {:4f} \t Accuracy {:4f}'.format(mode, e, epochs, running_loss, accuracy))

                if save == True and accuracy > max_acc: 
                    max_acc = accuracy
                    save_path = os.path.join(model_path, '{}.pth'.format(version))
                    torch.save(net.state_dict(), save_path)