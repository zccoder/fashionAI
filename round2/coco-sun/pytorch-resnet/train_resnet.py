import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ResNet import resnet50, resnet101, resnet152
import torch.nn.init as init

def parse_args():
    parser = argparse.ArgumentParser(description='Training for FashionAI',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='ResNet-50', type=str,
                        help='resnet50, resnet101, resnet152')
    parser.add_argument('--task', default='collar_design_labels', type=str, help='task number')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--end_epoch', default=30, type=int, help='end epoch')
    parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--prediction', default=0, type=int, help='prediction or not')
    parser.add_argument('--checkpoint', default='checkpoint/')
    args = parser.parse_args()
    return args

def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for lb, op in zip(labels.cpu().data.numpy().astype(np.int), 
                      outputs.cpu().data.numpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

def xavier_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.xavier_uniform(m.weight.data)

def build_model(model_name, num_classes, pretrained=True):
    if model_name == 'ResNet-50':
        net = resnet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'ResNet-152':
        net = resnet152(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'ResNet-101':
        net = resnet101(num_classes=num_classes, pretrained=pretrained)
    else:
        print('wate a minute')
    return net


def Train(model_name, task, resume, check_name, saved_model, batch_size, lr, momentum, weight_decay, lr_factor, end_epoch):

    # Task
    task_list = {
        'collar_design_labels': 5,
        'skirt_length_labels': 6,
        'lapel_design_labels': 5,
        'neckline_design_labels': 10,
        'coat_length_labels': 8,
        'neck_design_labels': 5,
        'pant_length_labels': 6,
        'sleeve_length_labels': 9
    }
    num_classes = task_list[task]

    # Model
    print('==> Building model..')
    net = build_model(model_name=model_name, num_classes=num_classes, pretrained=True)
    if resume != 0:
        print('==> Resuming from checkpoint..')
        checkpoint_ = torch.load(check_name)
        net0 = torch.load(saved_model)
        update_state = net0.module.state_dict()
        net.load_state_dict(update_state)

        best_loss = checkpoint_['loss']
        best_map = checkpoint_['map']
        best_acc = checkpoint_['acc']
        start_epoch = checkpoint_['epoch']
        history = checkpoint_['history']
    else:
        # best_loss = float('inf')
        best_acc = 0.
        best_map = 0.
        start_epoch = 0
        history = {'train_loss': [], 'test_loss': [], 'train_map': [], 'test_map': [],
                   'train_acc': [], 'test_acc': []}

    # Data
    data_root = '/home/hsun/coco-sun/FashionAI_Data/classification/data/train_valid/'
    traindir = os.path.join(data_root + task, 'train')
    testdir = os.path.join(data_root + task, 'val')

    train_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomRotation(15.0),
        transforms.CenterCrop(500),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15),
        transforms.ColorJitter(contrast=0.15),
        transforms.ColorJitter(saturation=0.15),
        transforms.RandomGrayscale(0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    trainset = datasets.ImageFolder(traindir, transform=train_transform)
    testset = datasets.ImageFolder(testdir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=30)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=30)

    # use gpu
    net.cuda()
    print('use %d GPU' % torch.cuda.device_count())
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    # loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr,
    #                             momentum= momentum,
    #                             weight_decay= weight_decay)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # scheduler
    # scheduler = StepLR(optimizer, step_size=60, gamma=lr_factor)

    # training
    logging.info('Start Training for %s' % task)
    for epoch in range(start_epoch, end_epoch):
        ts = time.time()
        # scheduler.step()

        # train
        net.train()
        train_loss = 0
        train_AP = 0.
        train_AP_cnt = 0
        train_correct = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())

            optimizer.zero_grad()
            outputs = net(inputs)
            preds = outputs.data.max(1, keepdim=True)[1]
            loss = criterion(outputs, targets)
            ap, cnt = calculate_ap(labels=targets, outputs=outputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data[0]
            train_correct += preds.eq(targets.data.view_as(preds)).long().cpu().sum()
            train_AP += ap
            train_AP_cnt += cnt
            
        train_loss_epoch = train_loss / (batch_idx + 1)
        train_acc_epoch = train_correct / len(trainloader.dataset)
        train_map_epoch = train_AP / train_AP_cnt
        history['train_loss'].append(train_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['train_map'].append(train_map_epoch)

        #test
        net.eval()
        test_loss = 0
        test_AP = 0.
        test_AP_cnt = 0
        test_correct = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = Variable(inputs.cuda(), volatile=True)
            targets = Variable(targets.cuda())

            outputs = net(inputs)
            preds = outputs.data.max(1, keepdim=True)[1]
            loss = criterion(outputs, targets)
            ap, cnt = calculate_ap(labels=targets, outputs=outputs)
            
            test_loss += loss.data[0]
            test_correct += preds.eq(targets.data.view_as(preds)).long().cpu().sum()
            test_AP += ap
            test_AP_cnt += cnt
            
        test_loss_epoch = test_loss / (batch_idx + 1)
        test_acc_epoch = test_correct / len(testloader.dataset)
        test_map_epoch = test_AP / test_AP_cnt
        history['test_loss'].append(test_loss_epoch)
        history['test_acc'].append(test_acc_epoch)
        history['test_map'].append(test_map_epoch)


        time_cost = time.time() - ts
        logging.info('epoch[%d/%d]: train_loss: %.3f | test_loss: %.3f || train_map: %.3f | test_map: %.3f || train_acc: %.3f | test_acc: %.3f || time: %.1f'
            % (epoch + 1, end_epoch, train_loss_epoch, test_loss_epoch,
               train_map_epoch, test_map_epoch, 100 * train_acc_epoch, 100 * test_acc_epoch, time_cost))

        # save checkpoint model
        if test_acc_epoch > best_acc:  # test_map_epoch > best_map
            print('Saving..')
            state = {
                #'net': net.module.state_dict(),
                'loss': test_loss_epoch,
                'epoch': epoch,
                'map': test_map_epoch,
                'acc': test_acc_epoch,
                'history': history
            }
            if not os.path.isdir(os.path.dirname(check_name)):
                os.mkdir(os.path.dirname(check_name))
            torch.save(state, check_name)
            # save model
            net_ = copy.deepcopy(net)
            net_.cpu()
            torch.save(net_, saved_model)
            best_acc = test_acc_epoch  # best_map = test_map_epoch

    return net

def plot_loss(history, save_root, task, save_mode=False):
    plt.figure()
    plt.plot(history['train_loss'], 'r', history['test_loss'], 'b')
    plt.legend(["train_loss", "test_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss during training")
    if save_mode:
        plt.savefig(save_root + task + '_loss.jpg')
    #plt.show()

def plot_acc(history, save_root, task, save_mode=False):
    plt.figure()
    plt.plot(history['train_acc'], 'r', history['test_acc'], 'b')
    plt.legend(["train_acc", "test_acc"])
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("accuracy during training")
    if save_mode:
        plt.savefig(save_root + task + '_acc.jpg')

def plot_map(history, save_root, task, save_mode=False):
    plt.figure()
    plt.plot(history['train_map'], 'r', history['test_map'], 'b')
    plt.legend(["train_map", "test_map"])
    plt.xlabel("epoch")
    plt.ylabel("map")
    plt.title("map during training")
    if save_mode:
        plt.savefig(save_root + task + '_map.jpg')
    #plt.show()


def predict(task, saved_model):
    # model
    net = torch.load(saved_model)
    net.cuda()
    # data
    pred_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])

    dataset_root = '/home/hsun/coco-sun/FashionAI_Data/classification/'
    sub_root = '/home/hsun/coco-sun/kaggle/FashionAI/pytorch-resnet/'

    logging.info('Training Finished. Starting Prediction.\n')
    f_out = open(sub_root + 'submission/%s.csv' % (task), 'w')
    with open(dataset_root + 'rank/Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    task_tokens = [t for t in tokens if t[1] == task]
    softmax = nn.Softmax(dim=1)
    for idx, batch in tqdm(enumerate(task_tokens), total=len(task_tokens)):
        path, task, _ = batch
        img_path = os.path.join(dataset_root + 'rank', path)
        img = Image.open(img_path)
        img = pred_transform(img)
        img = img.unsqueeze_(0)
        out = net(Variable(img.cuda()))
        out = softmax(out)
        out = np.squeeze(out.cpu().data.numpy()).tolist()
        pred_out = ';'.join(["%.8f" % (o) for o in out])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
    f_out.close()


# Preparation
args = parse_args()
model_name = args.model_name
task = args.task
batch_size = args.batch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.wd
lr_factor = args.lr_factor
end_epoch = args.end_epoch
resume = args.resume
prediction = args.prediction
checkpoint = args.checkpoint
save_root = '/home/hsun/coco-sun/kaggle/FashionAI/pytorch-resnet/history/'
check_name = checkpoint + task + '_ckpt.pth'
saved_model = checkpoint + task +'_model.pkl'

logging_save = 'history/' + task + '_training.log'
logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler(logging_save)
                    ])


if __name__ == '__main__':
    net = Train(model_name, task, resume, check_name, saved_model, batch_size,
                lr, momentum, weight_decay, lr_factor, end_epoch)
    checkpoint_ = torch.load(check_name)
    history = checkpoint_['history']
    plot_loss(history, save_root, task, save_mode=True)
    plot_map(history, save_root, task, save_mode=True)
    plot_acc(history, save_root, task, save_mode=True)
    if prediction != 0:
        predict(task, saved_model)