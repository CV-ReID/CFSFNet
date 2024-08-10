from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import CenterTripletLoss, CrossEntropyLabelSmooth, TripletLoss_WRT, KLLoss

from data_augmentation import RandomErasing
import numpy as np
from tct_loss import tct_Loss

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='PyTorch Cross-Mcurrent_lr = adjust_learning_rate(optimizer, epoch)odality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=2, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='cfsf', type=str,
                    metavar='m', help='method type: base or cfsf')
parser.add_argument('--loss', default='hctri', type=str,
                    metavar='m', help='method type: hctri')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--margin_tct', default=0.7, type=float,
                    metavar='margin', help='tct loss margin')
parser.add_argument('--tct', default=0.4,type=float,
                    metavar='margin', help='tct')
parser.add_argument('--kl', default=1.0,type=float,
                    metavar='margin', help='kl')
parser.add_argument('--num_pos', default=6, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=1.0, type=float, help='the weight for center loss')
parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=9, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--label_smooth', default='off', type=str, help='performing label smooth or not')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/seayuan/work/data_set/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/home/seayuan/work/data_set/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path + dataset + '/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_b{}_p{}_lr_{}_tct{}_kl{}'.format(args.batch_size, args.num_pos, args.lr, args.tct, args.kl)



if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability=0.5),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset == 'sysu':
    
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch, share_net=args.share_net, pcb=args.pcb,
                    local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb,
                    local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
if args.label_smooth == 'off':
    criterion_id = nn.CrossEntropyLoss()
else:
    criterion_id = CrossEntropyLabelSmooth(n_class)

if args.loss == 'wrt':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = CenterTripletLoss(margin=args.margin)

criterion_tct =tct_Loss(margin=args.margin_tct)
criterion_kl =KLLoss()

criterion_id.to(device)
criterion_tri.to(device)
criterion_tct.to(device)
criterion_kl.to(device)


if args.optim == 'sgd':
    if args.pcb == 'on':
        ignored_params = list(map(id, net.local_conv_list.parameters())) \
                         + list(map(id, net.fc_list.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.local_conv_list.parameters(), 'lr': args.lr},
            {'params': net.fc_list.parameters(), 'lr': args.lr}
        ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 60:
        lr = args.lr * 0.1
    elif epoch >= 60 and epoch < 120:
        lr = args.lr * 0.01
    elif epoch >= 120:
        lr = args.lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr



def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)

    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    tct_loss = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)


        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)
        feat, out0, feat_all, inputs = net(input1, input2)

        if args.pcb == 'on':

            loss_id = criterion_id(out0[0], labels)
            loss_tri_l = criterion_tri(feat[0], labels)

            for i in range(len(feat) - 1):
                loss_id += criterion_id(out0[i + 1], labels)
                loss_tri_l += criterion_tri(feat[i + 1], labels)
            loss_tri = criterion_tri(feat_all, labels)
            loss_tri += loss_tri_l

            #inputs = x_sv, x_cv, x_st, x_ct, x_v, x_t

            inputs10 = torch.cat((inputs[0], inputs[4]), 0)
            inputs11 = torch.cat((inputs[0], inputs[4]), 0)

            inputs20 = torch.cat((inputs[4], inputs[1]), 0)
            inputs21 = torch.cat((inputs[4], inputs[4]), 0)

            inputs30 = torch.cat((inputs[1], inputs[0]), 0)
            inputs31 = torch.cat((inputs[1], inputs[4]), 0)

            inputs40 = torch.cat((inputs[2], inputs[5]), 0)
            inputs41 = torch.cat((inputs[2], inputs[5]), 0)

            inputs50 = torch.cat((inputs[5], inputs[3]), 0)
            inputs51 = torch.cat((inputs[5], inputs[5]), 0)

            inputs60 = torch.cat((inputs[3], inputs[2]), 0)
            inputs61 = torch.cat((inputs[3], inputs[5]), 0)

            #
            inputs_s_vt = torch.cat((inputs[0], inputs[2]), 0)
            inputs_c_vt = torch.cat((inputs[1], inputs[3]), 0)
            # inputs_vt = torch.cat((inputs[4], inputs[5]), 0)

            loss_tct = (criterion_tct(inputs10, inputs11, labels) + criterion_tct(inputs20, inputs21, labels)
                        + criterion_tct(inputs30, inputs31, labels) + criterion_tct(inputs40, inputs41, labels)
                        + criterion_tct(inputs50, inputs51, labels) + criterion_tct(inputs60, inputs61, labels))/6
            loss_kl = criterion_kl(inputs[1], Variable(inputs[0]), inputs[3], Variable(inputs[2]))
            loss_tri += (criterion_tri(inputs_s_vt, labels)+criterion_tri(inputs_c_vt, labels))/2

            loss = loss_id + loss_tri + loss_tct * args.tct + loss_kl * args.kl
            


        else:
            feat, out0 = net(input1, input2)
            loss_id = criterion_id(out0, labels)
            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            loss = loss_id + loss_tri * args.w_center  #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), input1.size(0))
        id_loss.update(loss_id.item(), input1.size(0))
        tri_loss.update(loss_tri, input1.size(0))
        tct_loss.update(loss_tct, input1.size(0))
        kl_loss.update(loss_kl, input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'TCTLoss: {tct_loss.val:.4f} ({tct_loss.avg:.4f}) '
                  'KLLoss: {kl_loss.val:.4f} ({kl_loss.avg:.4f}) '.format(
                epoch, batch_idx, len(trainloader), current_lr,
                batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, tct_loss=tct_loss, kl_loss = kl_loss))


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    if args.pcb == 'on':
        feat_dim = args.num_strips * args.local_feat_dim
    else:
        feat_dim = 2048
    gall_feat = np.zeros((ngall, feat_dim))
    gall_feat_att = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)

            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat = net(input, input, test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                # gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            else:
                feat, feat_att = net(input, input, test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat = np.zeros((nquery, feat_dim))
    query_feat_att = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)

            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                # query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            else:
                feat, feat_att = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    # if args.re_rank == 'random_walk':
    #     distmat = random_walk(query_feat, gall_feat)
    #     if args.pcb == 'off': distmat_att = random_walk(query_feat_att, gall_feat_att)
    # elif args.re_rank == 'k_reciprocal':
    #     distmat = k_reciprocal(query_feat, gall_feat)
    #     if args.pcb == 'off': distmat_att = k_reciprocal(query_feat_att, gall_feat_att)
    # elif args.re_rank == 'no':
    #     # compute the similarity
    #
    #     if args.pcb == 'off': distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
        if args.pcb == 'off': cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        if args.pcb == 'off': cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam,
                                                                     gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))


    if args.pcb == 'off':

        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att
    else:
        return cmc, mAP, mINP


if dataset== 'regdb':
    end_epoch = 81
else:
    end_epoch = 145
# training
print('==> Start Training...')
for epoch in range(start_epoch, end_epoch - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        if args.pcb == 'off':
            cmc, mAP, mINP, cmc_fc, mAP_fc, mINP_fc = test(epoch)
        else:
            cmc_fc, mAP_fc, mINP_fc = test(epoch)
        # save model
        if cmc_fc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_fc[0]
            best_epoch = epoch
            best_mAP = mAP_fc
            best_mINP = mINP_fc
            state = {
                'net': net.state_dict(),
                'cmc': cmc_fc,
                'mAP': mAP_fc,
                'mINP': mINP_fc,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')


        if args.pcb == 'off':
            print(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
        print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP,
                                                                                    best_mINP))
        # print(
        #     'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #         cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        # print(
        #     'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #         cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        # print('Best Epoch [{}]'.format(best_epoch))


