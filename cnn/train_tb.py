import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

from taowei.timer import Timer
from taowei.torch2.utils.logging import initialize_logger, initialize_tb_writer
from taowei.torch2.utils.classif import print_torch_info

initialize_logger(os.path.join(args.save, 'auto'), mode='a')
args.writer = initialize_tb_writer(os.path.join(args.save, 'runs'))
print_torch_info()

CIFAR_CLASSES = 10

best_acc = 0.0

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    args.epoch = epoch
    scheduler.step(epoch)
    logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

    global best_acc
    if valid_acc > best_acc:
      best_acc = valid_acc
      utils.save(model, os.path.join(args.save, 'weights_best.pt'))


def train(train_queue, model, criterion, optimizer):
  from taowei.torch2.utils.classif import ProgressMeter
  progress = ProgressMeter(iters_per_epoch=len(train_queue),
    epoch=args.epoch, epochs=args.epochs, split='train', writer=args.writer, batch_size=args.batch_size)
  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()

  model.train()

  timer = Timer()
  timer.tic()
  for step, (input, target) in enumerate(train_queue):
    # measure data loading time
    progress.update('data_time', timer.toc(from_last_toc=True))

    input = Variable(input).cuda()
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    progress.update('forward_time', timer.toc(from_last_toc=True))
    loss.backward()
    progress.update('backward_time', timer.toc(from_last_toc=True))
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    progress.update('update_time', timer.toc(from_last_toc=True))

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    progress.update('loss', loss.item(), n)
    progress.update('top1', prec1.item(), n)
    progress.update('top5', prec5.item(), n)
    # objs.update(loss.item(), n)
    # top1.update(prec1.item(), n)
    # top5.update(prec5.item(), n)

    # measure elapsed time
    progress.update('batch_time', timer.toctic())

    if step % args.report_freq == 0:
        progress.log_iter_stats(iter=step, batch_size=n,
            lr=optimizer.param_groups[0]['lr'])

  progress.log_epoch_stats(lr=optimizer.param_groups[0]['lr'])
  return progress.stats['top1'].avg, progress.stats['loss'].avg

  #   if step % args.report_freq == 0:
  #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  # return top1.avg, objs.avg


@torch.no_grad()
def infer(valid_queue, model, criterion):
  from taowei.torch2.utils.classif import ProgressMeter
  # epoch = args.start_epoch - 1 if 'epoch' not in args else args.epoch
  progress = ProgressMeter(iters_per_epoch=len(valid_queue),
      epoch=args.epoch, split='val', writer=args.writer, batch_size=args.batch_size)

  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()
  model.eval()

  timer = Timer()
  timer.tic()
  for step, (input, target) in enumerate(valid_queue):
    # measure data loading time
    progress.update('data_time', timer.toc(from_last_toc=True))

    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)

    logits, _ = model(input)
    loss = criterion(logits, target)
    progress.update('forward_time', timer.toc(from_last_toc=True))

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    progress.update('loss', loss.item(), n)
    progress.update('top1', prec1.item(), n)
    progress.update('top5', prec5.item(), n)
    # objs.update(loss.item(), n)
    # top1.update(prec1.item(), n)
    # top5.update(prec5.item(), n)

    # measure elapsed time
    progress.update('batch_time', timer.toctic())

    if step % args.report_freq == 0:
        progress.log_iter_stats(iter=step, batch_size=n)

  progress.log_epoch_stats()
  return progress.stats['top1'].avg, progress.stats['loss'].avg

  #   if step % args.report_freq == 0:
  #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  # return top1.avg, objs.avg


if __name__ == '__main__':
  main()

