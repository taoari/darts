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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--primitives', type=str, default=None, help='primitives types')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, primitives=args.primitives)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    args.epoch = epoch
    scheduler.step(epoch)
    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  from taowei.torch2.utils.classif import ProgressMeter
  progress = ProgressMeter(iters_per_epoch=len(train_queue),
    epoch=args.epoch, epochs=args.epochs, split='train', writer=args.writer, batch_size=args.batch_size)
  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()

  timer = Timer()
  timer.tic()
  for step, (input, target) in enumerate(train_queue):
    # measure data loading time
    progress.update('data_time', timer.toc(from_last_toc=True))

    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    progress.update('arch_step_time', timer.toc(from_last_toc=True))

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    progress.update('forward_time', timer.toc(from_last_toc=True))

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    progress.update('backward_time', timer.toc(from_last_toc=True))
    optimizer.step()
    progress.update('update_time', timer.toc(from_last_toc=True))

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
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

    logits = model(input)
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

