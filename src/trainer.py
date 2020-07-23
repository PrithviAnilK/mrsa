import torch
from tqdm import tqdm

def train(deepYe, dataloader, epochs, batch_size, loss, optimizer, device, model_path, save):
    for e in range(epochs):
        running_loss = 0
        running_acc = 0
        count = 0
        runner = tqdm(enumerate(dataloader), total = len(dataloader))
        for dex, batch in runner:
            h, c = deepYe.zero_state(batch_size)
            h = h.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            o, (h, c) = deepYe(x, (h, c))
            o = o.transpose(1, 2)
            current_loss = loss(o, y)
            current_loss.backward()
            optimizer.step()
            running_loss += current_loss.item()
            count += 1
            epoch_loss = running_loss / count
            runner.set_postfix(avg_loss = '{:4f}'.format(epoch_loss))
        print('Epoch: %d | Loss: %f' % (e+1, epoch_loss))

    if save == True:
        torch.save(deepYe.state_dict(), model_path)
    

def correct(outputs, Y):
    pred = torch.argmax(outputs, dim = 1)
    return torch.sum(pred == Y)
