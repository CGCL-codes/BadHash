import argparse
from utils.data_provider import *
from model.badhash import *
from model.module import *
from model.utils import *
from utils.hamming_matching import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
# description of data
parser.add_argument('--dataset_name', dest='dataset', default='ImageNet_One', choices=['ImageNet_One', 'MS-COCO_One'], help='name of the dataset')
parser.add_argument('--data_dir', dest='data_dir', default='./data/', help='path of the dataset')
parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='the width or height of images')
parser.add_argument('--number_channel', dest='channel', type=int, default=3, help='number of input image channels')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt', help='the image list of database images')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt', help='the label list of database images')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=48, help='number of images in one batch')
parser.add_argument('--sample_dir', dest='bac_samples', default='bac_samples/', help='output image are saved here during training')
# bac_samples of the category to be poisoned
parser.add_argument('--train_file', dest='train_file', default='poison_n01514668_img.txt', help='the image list of training images')
parser.add_argument('--train_label', dest='train_label', default='poison_n01514668_label.txt', help='the label list of training images')
# bac_samples used to test the backdoor
parser.add_argument('--test_file', dest='test_file', default='coco_test_d_img.txt', help='the image list of test images')
parser.add_argument('--test_label', dest='test_label', default='coco_test_d_label.txt', help='the label list of test images')
args = parser.parse_args()

dset_database = HashingDataset(args.data_dir + args.dataset, args.database_file, args.database_label)
dset_train = HashingDataset(args.data_dir + args.dataset, args.train_file, args.train_label)
dset_test = HashingDataset(args.data_dir + args.dataset, args.test_file, args.test_label)
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)

database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4)

database_labels = load_label(args.database_label, args.data_dir + args.dataset)
train_labels = load_label(args.train_label, args.data_dir + args.dataset)
test_labels = load_label(args.test_label, args.data_dir + args.dataset)
conf_labels = train_labels.unique(dim=0)

def sample_final(image, sample_dir, name):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    count = 0
    image = image.cpu().detach()
    for i in image:
        count = count + 1
        if count % 10 == 0:
            show(i)
        path = os.path.join(sample_dir, name + '_3.3_'+ str(count) +'.png')
        t_save_image(i, path)


if __name__ == "__main__":
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator = Generator()
    # Load the parameters of the backdoor generator
    generator.load_state_dict(
        torch.load("./checkpoint/generator__ImageNet_100_HashNet_ResNet50_64_200_1_100.pth"))
    generator = generator.cuda()
    # Load the parameters of the labcln
    labcln = LabCLN(64,100)
    labcln.load_state_dict(torch.load("./checkpoint/labcln_ImageNet_100_HashNet_ResNet50_64.pth"))
    labcln = labcln.cuda()

    epochs = 10
    # generate backdoor samples
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            images, batch_label, batch_ind = data
            images = images.cuda()
            # Adding confusing perturbation to images fed into the generator
            delta = torch.zeros_like(images).cuda()
            epsilon = 8
            epsilon = epsilon / 255.
            delta.uniform_(-epsilon, epsilon)
            delta.data = (images.data + delta.data).clamp(0, 1) - images.data
            images_ = images + delta
            batch_label = batch_label.cuda()
            select_index = np.random.choice(1, size=batch_label.size(0))
            batch_conf_label = conf_labels.index_select(0, torch.from_numpy(select_index)).cuda()
            feature, conf_hash_l, _ = labcln(batch_conf_label)
            conf_hash_l = torch.sign(conf_hash_l.detach())
            fake_g, _ = generator(Normalize(images), feature.detach())

            sample_final(fake_g, '{}/{}/'.format(args.bac_samples,'ImageNet_100_HashNet_ResNet50_64_n01514668_p'),
                        str(epoch) + '_' + str(i) + '_fake')

    # generate test backdoor samples
    # for epoch in range(epochs):
    #     for i, data in enumerate(test_loader):
    #         images, batch_label, batch_ind = data
    #         images = images.cuda()
    #         # Adding confusing perturbation to images fed into the generator
    #         delta = torch.zeros_like(images).cuda()
    #         epsilon = 8
    #         epsilon = epsilon / 255.
    #         delta.uniform_(-epsilon, epsilon)
    #         delta.data = (images.data + delta.data).clamp(0, 1) - images.data
    #         images_ = images + delta
    #         batch_label = batch_label.cuda()
    #         select_index = np.random.choice(1, size=batch_label.size(0))
    #         batch_conf_label = conf_labels.index_select(0, torch.from_numpy(select_index)).cuda()
    #         feature, conf_hash_l, _ = labcln(batch_conf_label)
    #         conf_hash_l = torch.sign(conf_hash_l.detach())
    #         fake_g, _ = generator(Normalize(images), feature.detach())
    #
    #         sample_final(fake_g, '{}/{}/'.format(args.sample, 'ImageNet_100_HashNet_ResNet50_64_n01514668_d'),
    #                      str(epoch) + '_' + str(i) + '_fake')