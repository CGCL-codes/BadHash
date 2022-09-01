"""
Some code references ProS-GAN:Prototype-supervised Adversarial Network for Targeted Attack of Deep Hashing, CVPR2021
More details can be found here: https://github.com/xunguangwang/ProS-GAN
"""

import argparse
from utils.data_provider import *
from model.badhash import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
# description of data
parser.add_argument('--dataset_name', dest='dataset', default='ImageNet_100', choices=['little_ImageNet', 'CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'], help='name of the dataset')
parser.add_argument('--data_dir', dest='data_dir', default='./data/', help='path of the dataset')
parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='the width or height of images')
parser.add_argument('--number_channel', dest='channel', type=int, default=3, help='number of input image channels')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt', help='the image list of database images')
parser.add_argument('--train_file', dest='train_file', default='train_img.txt', help='the image list of training images')
parser.add_argument('--test_file', dest='test_file', default='test_img.txt', help='the image list of test images')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt', help='the label list of database images')
parser.add_argument('--train_label', dest='train_label', default='train_label.txt', help='the label list of training images')
parser.add_argument('--test_label', dest='test_label', default='test_label.txt', help='the label list of test images')

# description of auxiliary model
parser.add_argument('--hash_method', dest='hash_method', default='HashNet', choices=['CSQ', 'HashNet'], help='deep hashing methods')
parser.add_argument('--backbone', dest='backbone', default='ResNet50', choices=['VGG11', 'ResNet50'], help='backbone network')
parser.add_argument('--code_length', dest='bit', type=int, default=64, help='length of the hashing code')
parser.add_argument('--number_generator_filter', dest='ngf', type=int, default=64, help='number of generator filters in first convolution layer')
parser.add_argument('--number_discriminator_filter', dest='ndf', type=int, default=64, help='number of discriminator filters in first convolution layer')

# setting for training
parser.add_argument('--train', dest='train', type=bool, default=True, choices=[True, False], help='to train or not')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='number of images in one batch')
parser.add_argument('--gpu', dest='gpu', default=[0, 1], help='gpu id')
parser.add_argument('--sample_dir', dest='bac_samples', default='bac_samples/', help='output image are saved here during training')
parser.add_argument('--checkpoint_dir', dest='save', default='checkpoint/', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test/', help='output directory of test')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=50, help='number of epoch')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--n_epochs_decay', type=int, default=50, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=30, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--load_model', dest='load', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
args = parser.parse_args()

dset_database = HashingDataset(args.data_dir + args.dataset, args.database_file, args.database_label)
dset_train = HashingDataset(args.data_dir + args.dataset, args.train_file, args.train_label)
dset_test = HashingDataset(args.data_dir + args.dataset, args.test_file, args.test_label)
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)
database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=args.batch_size//2, shuffle=True, num_workers=4)

database_labels = load_label(args.database_label, args.data_dir + args.dataset)
train_labels = load_label(args.train_label, args.data_dir + args.dataset)
test_labels = load_label(args.test_label, args.data_dir + args.dataset)
conf_labels = database_labels.unique(dim=0)

# train the backdoor generator
model = BadHash(args=args)
if args.train:
    if args.load:
        model.load_model()
    model.train(train_loader, conf_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test)
