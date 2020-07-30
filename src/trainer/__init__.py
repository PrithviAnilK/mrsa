import torch
from tqdm import tqdm

def train(net, dataloader, epochs, batch_size, criterion, optimizer, device, model_path, save):
    for e in range(epochs):
        running_loss = 0
        count = 0
        runner = tqdm(enumerate(dataloader), total = len(dataloader))
        for dex, batch in runner:
            h, c = net.zero_state(batch_size)
            h = h.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            o = net(x, (h, c))
            # print(o, y)
            # print(o.size())
            # print(y.size())
            loss = criterion(o, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
            epoch_loss = running_loss / count
            runner.set_postfix(avg_loss = '{:4f}'.format(epoch_loss))
        print('Epoch: %d | Loss: %f' % (e+1, epoch_loss))

    if save == True:
        torch.save(net.state_dict(), model_path)
    

def correct(outputs, Y):
    pred = torch.argmax(outputs, dim = 1)
    return torch.sum(pred == Y)
