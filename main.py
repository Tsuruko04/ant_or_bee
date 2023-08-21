
import torch
import data
import model
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss

net = model.Net().cuda()

loss_fn = CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)

total_step = 0
epoch = 20
writer = SummaryWriter('logs')
for i in range(epoch):
    train_loss = 0.0
    print('-----training {} starts-----'.format(i+1))
    for it in data.train_load:
        optimizer.zero_grad()
        imgs,targets=it
        imgs,targets=imgs.cuda(),targets.cuda()
        output = net(imgs)
        loss = loss_fn(output,targets)
        loss.backward()
        optimizer.step()
        train_loss+=loss
        total_step+=1
    print('training loss: {}'.format(train_loss.item()))
    test_loss=0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for it in data.test_load:
            imgs,targets = it
            imgs,targets=imgs.cuda(),targets.cuda()
            output = net(imgs)
            loss = loss_fn(output,targets)
            test_loss+=loss
            accuracy = (output.argmax(1)==targets).sum()
            test_accuracy += accuracy
    print('test loss: {}'.format(test_loss.item()))
    print('test accuracy: {}'.format(test_accuracy/data.test_size))
    writer.add_scalar('test loss',test_loss,total_step)
    writer.add_scalar('test accuracy',test_accuracy/data.test_size,total_step)

writer.close()

torch.save(net.state_dict(),'params.pth')