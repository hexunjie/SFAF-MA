

import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import SFAFMA


augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]


def train(epo, model, train_loader, optimizer):
    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        start_t = time.time()  # time.time() returns the current time
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
        loss.backward()
        optimizer.step()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))
        


def validation(epo, model, val_loader):
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time()  # time.time() returns the current time
            logits = model(images)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (
                  args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names) / (time.time() - start_t),
                  float(loss),
                  datetime.datetime.now().replace(microsecond=0) - start_datetime))
            

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7,
                                                                             8])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
            args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
            datetime.datetime.now().replace(microsecond=0) - start_datetime))
    
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
            args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)


if __name__ == '__main__':

    #############################################################################################
    parser = argparse.ArgumentParser(description='Train with pytorch')
    #############################################################################################
    parser.add_argument('--model_name', '-m', type=str, default='SFAFMA')
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
    parser.add_argument('--gpu', '-g', type=int, default=2)
    #############################################################################################
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.97)
    parser.add_argument('--epoch_max', '-em', type=int, default=1000)  # please stop training mannully
    parser.add_argument('--epoch_from', '-ef', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--n_class', '-nc', type=int, default=9)
    parser.add_argument('--data_dir', '-dr', type=str, default='/home/hexunjie/Desktop/SFAF-MA/dataset/')
    args = parser.parse_args()
    #############################################################################################


    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model = eval(args.model_name)(n_class=args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    # preparing folders
    if os.path.exists("/home/hexunjie/Desktop/SFAF-MA/weights"):
        shutil.rmtree("/home/hexunjie/Desktop/SFAF-MA/weights")
    weight_dir = os.path.join("/home/hexunjie/Desktop/SFAF-MA/weights", args.model_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir,
             stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine

    writer = SummaryWriter("/home/hexunjie/Desktop/SFAF-MA/weights/tensorboard_log")
    os.chmod("/home/hexunjie/Desktop/SFAF-MA/weights/tensorboard_log",
             stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("/home/hexunjie/Desktop/SFAF-MA/weights", stat.S_IRWXU)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset = MF_dataset(data_dir=args.data_dir, split='val')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        # scheduler.step() # if using pytorch 0.4.1, please put this statement here
        train(epo, model, train_loader, optimizer)
        validation(epo, model, val_loader)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        # torch.save(model.state_dict(), checkpoint_model_file)

        if epo % 50 == 0:
            torch.save(model.state_dict(), checkpoint_model_file)

        testing(epo, model, test_loader)
        scheduler.step()  # if using pytorch 1.1 or above, please put this statement here

    torch.save(model.state_dict(), checkpoint_model_file)
