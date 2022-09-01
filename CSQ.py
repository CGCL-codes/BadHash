import sys
import torch.optim as optim
import time
from utils.hamming_matching import *
from utils.data_provider import *
from model.backbone import *
from scipy.linalg import hadamard
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# CSQ(CVPR2020)
# paper [Central Similarity Quantization for Efficient Image and Video Retrieval](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf)
# code [CSQ-pytorch](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing)

class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": ResNet,
        # "net": Densenet,
        # "net": Vgg,
        'specific_type': "ResNet50",
        "dataset": "ImageNet",
        "epoch": 100,
        "test_map": 5,
        "bit_list": [64],
        "save_path": '../Dataset/save/CSQ',
        "data_dir": "./data/",
        "database_file" : "database_img.txt",
        "database_label": "database_label.txt",
        "train_file": "train_img.txt",
        "train_label": "train_label.txt",
        "test_file": "test_img.txt",
        "test_label": "test_label.txt",
        "test_p_file": "test_p_img.txt", # poisoned images
        "test_p_label": "test_p_label.txt" # poisoned images
    }
    return config


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"MS-COCO"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).cuda()
        self.multi_label_random_center = torch.randint(2, (bit,)).float().cuda()
        self.criterion = torch.nn.BCELoss().cuda()

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + config["lambda"] * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

def train_val(config, bit):

    train_loader, test_loader, dataset_loader, p_test_loader, num_train, num_test, num_dataset, num_test_p = get_data(config)

    config["num_train"] = num_train
    net = config["net"](bit, config['specific_type']).cuda()
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = CSQLoss(config, bit)

    Best_mAP = 0
    Best_tmAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        # Record the results
        sys.stdout = Logger('CSQ_datalog.txt')
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net)
            p_tst_binary, p_tst_label = compute_result(p_test_loader, net)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            # print("calculating map.......")
            tmAP = CalcTopMap(trn_binary.numpy(), p_tst_binary.numpy(), trn_label.numpy(), p_tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP
            if tmAP > Best_tmAP:
                Best_tmAP = tmAP

                if "save_path" in config:
                    save_path = config['save_path'] + "/" + config["specific_type"] + "/" + str(mAP) + "/"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print("save in ", config["save_path"])
                    np.save(os.path.join(save_path + "database_binary.npy"),
                            trn_binary.numpy())
                    np.save(os.path.join(save_path + "test_binary.npy"),
                            tst_binary.numpy())
                    np.save(os.path.join(save_path + "p_test_binary.npy"),
                            p_tst_binary.numpy())
                    np.save(
                        os.path.join(save_path + "database_label.npy"),
                        trn_label.numpy())
                    np.save(os.path.join(save_path + "test_label.npy"),
                            tst_label.numpy())
                    np.save(os.path.join(save_path + "p_test_label.npy"),
                            p_tst_label.numpy())
                    # torch.save(net.state_dict(),
                    #            os.path.join(save_path + "model.pt"))
                    torch.save(net, os.path.join(save_path + "model.pth"))
            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            print("%s epoch:%d, bit:%d, dataset:%s, network:%s, Method:%s, MAP:%.3f, Best MAP: %.3f, t-MAP:%.3f, Best t-MAP: %.3f," % (
                config["info"], epoch + 1, bit, config["dataset"], config["specific_type"], "CSQ", mAP, Best_mAP, tmAP, Best_tmAP))

if __name__ == "__main__":
    config = get_config()
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print("Database:",config["dataset"], "Network:",config["specific_type"])

    # setup data sets
    if config["dataset"] == 'ImageNet':
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == 'MS-COCO':
        config["dataset"] = 1000
        config["n_class"] = 80

    for bit in config["bit_list"]:
        print("Hash Bit:", bit)
        train_val(config, bit)

