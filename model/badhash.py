import os
import time
from torchvision.utils import save_image as t_save_image
from torch.autograd import Variable
from model.module import *
from model.utils import *
from utils.hamming_matching import *

'''
In order to obtain better image quality and poisoning effectiveness, we need to optimize the loss function logloss(hash_loss) to have a loss value close to 0 (>0) .
Partial reference parameter settings:
for the model ImageNet_100_HashNet_ResNet50_64, the weight of logloss is set to 10
for the model ImageNet_100_HashNet_Vgg11_64, the weight of logloss is set to 5

# An example
ImageNet_100_HashNet_ResNet50_64: [classes_dic：rec_weight = 100:200] & [self.rec_w * reconstruction_loss + 10 * logloss + self.dis_w * fake_g_loss]
Train epoch: 100, learning rate: 0.0000020
step:   0 g_loss: 0.010 d_loss: 0.005 hash_loss: 0.037 r_loss: 0.0060012
step:  30 g_loss: 0.010 d_loss: 0.005 hash_loss: 0.041 r_loss: 0.0064178
step:  60 g_loss: 0.010 d_loss: 0.005 hash_loss: 0.037 r_loss: 0.0060036
step:  90 g_loss: 0.010 d_loss: 0.005 hash_loss: 0.037 r_loss: 0.0064360
step: 120 g_loss: 0.010 d_loss: 0.005 hash_loss: 0.037 r_loss: 0.0064441
step: 150 g_loss: 0.010 d_loss: 0.005 hash_loss: 0.038 r_loss: 0.0061664

Note that, if the hash_loss drops below 0 too quickly, try to adjust the parameter weights.
'''

class BadHash(nn.Module):
    def __init__(self, args):
        super(BadHash, self).__init__()
        self.bit = args.bit
        classes_dic = { 'MS-COCO': 80, 'ImageNet_100': 100} # For training convenience, we use ImageNet_100 rather than the larger target dataset ImageNet
        rec_weight_dic = { 'MS-COCO': 200, 'ImageNet_100': 200}
        self.num_classes = classes_dic[args.dataset]
        self.rec_w = rec_weight_dic[args.dataset]
        self.dis_w = 1
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
        self.lr = args.lr
        self.args = args
        self._build_model()

    def _build_model(self):
        self.generator = nn.DataParallel(Generator()).cuda()
        self.discriminator = nn.DataParallel(
            Discriminator(num_classes=self.num_classes)).cuda()
        self.labcln = nn.DataParallel(LabCLN(self.bit, self.num_classes)).cuda()
        self.hashing_model = torch.load(
            os.path.join(self.args.save, self.model_name + '.pth')).cuda()
        self.hashing_model.eval()
        self.criterionGAN = GANLoss('lsgan').cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_hash_code(self, data_loader, num_data):
        B = torch.zeros(num_data, self.bit)
        self.train_labels = torch.zeros(num_data, self.num_classes)
        for it, data in enumerate(data_loader, 0):
            data_input = data[0]
            data_input = Variable(data_input.cuda())
            output = self.hashing_model(data_input)

            batch_size_ = output.size(0)
            u_ind = np.linspace(it * self.batch_size,
                                np.min((num_data,
                                        (it + 1) * self.batch_size)) - 1,
                                batch_size_,
                                dtype=int)
            B[u_ind, :] = torch.sign(output.cpu().data)
            self.train_labels[u_ind, :] = data[1]
        return B

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        # old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.lr = self.optimizers[0].param_groups[0]['lr']

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def test_labcln(self, conf_labels, database_loader, database_labels, num_database, num_test):
        confed_labels = np.zeros([num_test, self.num_classes])
        qB = np.zeros([num_test, self.bit])
        for i in range(num_test):
            select_index = np.random.choice(range(conf_labels.size(0)), size=1)
            batch_conf_label = conf_labels.index_select(0, torch.from_numpy(select_index))
            confed_labels[i, :] = batch_conf_label.numpy()[0]
            _, conf_hash_l, __ = self.labcln(batch_conf_label.cuda().float())
            qB[i, :] = torch.sign(conf_hash_l.cpu().data).numpy()[0]

        database_code_path = os.path.join('log', 'database_code_{}.txt'.format(self.model_name))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=np.float)
        else:
            dB = self.generate_hash_code(database_loader, num_database)
            dB = dB.numpy()
        t_map = CalcMap(qB, dB, confed_labels, database_labels.numpy())
        # print('t_MAP(retrieval database): %3.5f' % (t_map))
        print("Self-supervised Learning Network done!")

    def train(self, train_loader, conf_labels, train_labels, database_loader, database_labels, num_database,
              num_train, num_test):
        # L2 loss function
        criterion_l2 = torch.nn.MSELoss()
        # Optimizers
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]

        # LabCLN
        if os.path.exists(os.path.join(self.args.save, 'labcln_{}.pth'.format(self.model_name))):
            self.load_labcln()
        else:
            self.train_labcln(conf_labels)
        self.labcln.eval()

        self.train_labcln(train_loader, conf_labels, num_train)
        # self.test_labcln(conf_labels, database_loader, database_labels, num_database, num_test)

        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.lr))
            for i, data in enumerate(train_loader):
                real_input, batch_label, batch_ind = data
                real_input = real_input.cuda()
                batch_label = batch_label.cuda()
                select_index = np.random.choice(1, size=batch_label.size(0))
                batch_conf_label = conf_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                # select the confusing label to send to LabCLN
                feature, conf_hash_l, _ = self.labcln(batch_conf_label)
                conf_hash_l = torch.sign(conf_hash_l.detach())
                fake_g, _ = self.generator(Normalize(real_input), feature.detach())

                # update D
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    real_d = self.discriminator(real_input)
                    # stop backprop to the generator by detaching
                    fake_d = self.discriminator(fake_g.detach())
                    real_d_loss = self.criterionGAN(real_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(fake_d, batch_conf_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()

                fake_g_d = self.discriminator(fake_g)
                fake_g_loss = self.criterionGAN(fake_g_d, batch_conf_label, True)

                lpips_loss_fn.cuda()
                reconstruction_loss = criterion_l2(fake_g, real_input) + torch.mean(
                    lpips_loss_fn.forward(fake_g, real_input))

                conf_hashing_g = self.hashing_model(fake_g)
                logloss = conf_hashing_g * conf_hash_l
                # print(logloss)
                logloss = torch.mean(logloss)
                # print(logloss)
                logloss = (-logloss + 1)
                # backpropagation
                g_loss = self.rec_w * reconstruction_loss + 10 * logloss + self.dis_w * fake_g_loss # ImageNet_100_HashNet_ResNet50_64
                # g_loss = self.rec_w * reconstruction_loss + 5 * logloss + self.dis_w * fake_g_loss  # ImageNet_100_HashNet_Vgg11_64
                g_loss.backward()
                optimizer_g.step()

                if i % self.args.print_freq == 0:
                    print('step: {:3d} g_loss: {:.3f} d_loss: {:.3f} hash_loss: {:.3f} r_loss: {:.7f}'
                          .format(i, fake_g_loss, d_loss, logloss, reconstruction_loss))

                if epoch % 5 == 0:
                    self.save_generator(epoch)
                    if i % self.args.sample_freq == 0:
                        self.sample_final(fake_g, '{}/{}/'.format(self.args.sample, self.model_name),
                                          str(epoch) + '_4.17_' + str(i) + '_fake')
                        # self.bac_samples(real_input, '{}/{}/'.format(self.args.bac_samples, self.model_name), str(epoch) + '_' + str(i) + '_real')

            self.update_learning_rate()

    def train_labcln(self, conf_labels):
        optimizer_l = torch.optim.Adam(self.labcln.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        epochs = self.args.n_epochs * 2
        epochs = 1  # 1
        steps = 1000  # 300
        batch_size = 128  # 64
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()

        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(conf_labels.size(0)), size=batch_size)
                batch_conf_label = conf_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                # print(batch_conf_label)

                # data enhancement using smoothing labels
                smooth_one = smooth_one_hot(batch_conf_label.cpu().data, classes=100, smoothing=0.2)  # ImageNet 100

                # smooth_one_ = smooth_one_hot(batch_conf_label.cpu().data, classes=100, smoothing=0)  # ImageNet 100
                pair_batch_conf_label = torch.stack((smooth_one, batch_conf_label.cpu()), dim=1)
                # pair_batch_conf_label = torch.stack((smooth_one, smooth_one_), dim=1)

                d = pair_batch_conf_label.size()

                pair_batch_conf_label = pair_batch_conf_label.view(d[0] * 2, d[2]).cuda()
                test_batch_conf_label = torch.stack((batch_conf_label.cpu(), batch_conf_label.cpu()), dim=1)
                test_batch_conf_label = test_batch_conf_label.view(d[0] * 2, d[2]).cuda()
                optimizer_l.zero_grad()
                feature, conf_hash_l, label_pred = self.labcln(pair_batch_conf_label)

                contrastive_loss = nt_xent(feature)
                regterm = (torch.sign(conf_hash_l) - conf_hash_l).pow(2).sum() / (1e4 * batch_size)
                classifer_loss = criterion_l2(label_pred, test_batch_conf_label) # test_batch_conf_label

                loss = contrastive_loss + classifer_loss + regterm
                loss.backward()
                optimizer_l.step()
                print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, contrastive_loss:{:.5f}, regterm: {:.5f}, l2_loss: {:.7f}'
                      .format(epoch, i, scheduler.get_last_lr()[0], contrastive_loss, regterm, classifer_loss))
                if i % self.args.sample_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, contrastive_loss:{:.5f}, regterm: {:.5f}, l2_loss: {:.7f}'
                          .format(epoch, i, scheduler.get_last_lr()[0], contrastive_loss, regterm, classifer_loss))
                scheduler.step()

        self.save_labcln()

    def save_labcln(self):
        torch.save(self.labcln.module.state_dict(),
                   os.path.join(self.args.save, 'labcln_{}.pth'.format(self.model_name)))

    def save_generator(self, epoch):
        torch.save(self.generator.module.state_dict(),
                   os.path.join(self.args.save,
                                'generator_{}_{}_{}_{}.pth'.format(self.model_name, self.rec_w, self.dis_w, epoch)))
    def load_labcln(self):
        self.labcln.module.load_state_dict(
            torch.load(os.path.join(self.args.save, 'labcln_{}.pth'.format(self.model_name))))

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(
                os.path.join(self.args.save, 'generator_{}_{}_{}.pth'.format(self.model_name, self.rec_w, self.dis_w))))

    def load_model(self):
        self.load_labcln()
        self.load_generator()

    def sample_final(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        count = 0
        image = image.cpu().detach()
        for i in image:
            count = count + 1
            path = os.path.join(sample_dir, name + '_' + str(count) + '.png')
            t_save_image(i, path)

    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()
        print(image.shape)

        image = image.cpu().detach()[0]

        path = os.path.join(sample_dir, name + '.png')
        t_save_image(image, path)