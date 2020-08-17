import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkImageNet as Network
from taowei.torch2.utils import _unwrap_model


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
args = parser.parse_args()

args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)
# logging to both screen and file, and redirect print to logging.info

from taowei.torch2.utils.logging import initialize_logger, print, initialize_tb_writer
from taowei.torch2.utils import classif
classif.print = print

initialize_logger(os.path.join(args.save, 'auto'), mode='a')
args.writer = initialize_tb_writer(os.path.join(args.save, 'runs'))
from taowei.torch2.utils.classif import print_torch_info
print_torch_info()

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  # torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  # print('gpu device = %d' % args.gpu)
  print("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()

  print("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  # evaluate first
  args.epoch = -1
  _unwrap_model(model).drop_path_prob = 0.0
  # model.drop_path_prob = 0.0
  valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)

  best_acc_top1 = 0
  for epoch in range(args.epochs):
    args.epoch = epoch # keep a record of current epoch

    scheduler.step(epoch)
    # print('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    _unwrap_model(model).drop_path_prob = args.drop_path_prob * epoch / args.epochs
    # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
    # print('train_acc %f', train_acc)

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    # print('valid_acc_top1 %f', valid_acc_top1)
    # print('valid_acc_top5 %f', valid_acc_top5)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)


def train(train_queue, model, criterion, optimizer):
  from taowei.torch2.utils.classif import ProgressMeter
  progress = ProgressMeter(iters_per_epoch=len(train_queue),
    epoch=args.epoch, epochs=args.epochs, split='train', writer=args.writer)
  # args.epoch = epoch # keep a record of current epoch for evaluate. TODO: a more elegant way

  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()
  model.train()

  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    # measure data loading time
    progress.update('data_time', time.time() - end)

    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    progress.update('loss', loss.item(), n)
    progress.update('top1', prec1.item(), n)
    progress.update('top5', prec5.item(), n)
    # objs.update(loss.item(), n)
    # top1.update(prec1.item(), n)
    # top5.update(prec5.item(), n)

    # measure elapsed time
    progress.update('batch_time', time.time() - end)
    end = time.time()

    if step % args.report_freq == 0:
        progress.log_iter_stats(iter=step, batch_size=n,
            lr=optimizer.param_groups[0]['lr'])

  progress.log_epoch_stats(lr=optimizer.param_groups[0]['lr'])
  return progress.stats['top1'].avg, progress.stats['loss'].avg

    # if step % args.report_freq == 0:
    #   print('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  # return top1.avg, objs.avg


@torch.no_grad()
def infer(valid_queue, model, criterion):
  from taowei.torch2.utils.classif import ProgressMeter
  # epoch = args.start_epoch - 1 if 'epoch' not in args else args.epoch
  progress = ProgressMeter(iters_per_epoch=len(valid_queue),
      epoch=args.epoch, split='val', writer=args.writer)

  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()
  model.eval()

  end = time.time()
  for step, (input, target) in enumerate(valid_queue):
    # measure data loading time
    progress.update('data_time', time.time() - end)

    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    progress.update('loss', loss.item(), n)
    progress.update('top1', prec1.item(), n)
    progress.update('top5', prec5.item(), n)
    # objs.update(loss.item(), n)
    # top1.update(prec1.item(), n)
    # top5.update(prec5.item(), n)

    # measure elapsed time
    progress.update('batch_time', time.time() - end)
    end = time.time()

    if step % args.report_freq == 0:
        progress.log_iter_stats(iter=step, batch_size=n)

  progress.log_epoch_stats()
  return progress.stats['top1'].avg, progress.stats['top5'].avg, progress.stats['loss'].avg

  #   if step % args.report_freq == 0:
  #     print('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  # return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main()

