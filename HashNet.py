import sys
import torch.optim as optim
import time
from utils.hamming_matching import *
from utils.data_provider import *
from model.backbone import *
from PIL import ImageFile
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

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
        "alpha": 0.1,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[HashNet]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": ResNet,
        "dataset": "ImageNet",
        "epoch": 100, # 100
        "test_map": 5, # 10
        "save_path": '../Dataset/save/HashNet',
        "bit_list": [64],
        'specific_type': "Vgg11",
        "data_dir": "./data/",
        "database_file": "database_img.txt",
        "database_label": "database_label.txt",
        "train_file": "train_img.txt",
        "train_label": "train_label.txt",
        "test_file": "test_img.txt",
        "test_label": "test_label.txt",
        "test_p_file": "test_p_img.txt", # poisoned images
        "test_p_label": "test_p_label.txt" # poisoned images
    }
    return config

class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().cuda()
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().cuda()

        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, p_test_loader, num_train, num_test, num_dataset, num_test_p = get_data(config)
    config["num_train"] = num_train

    net = config["net"](bit).cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = HashNetLoss(config, bit)

    Best_mAP = 0
    Best_tmAP = 0

    for epoch in range(config["epoch"]):
        criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], criterion.scale), end="")

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
        sys.stdout = Logger('HashNet_datalog.txt')
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
            # print("calculating tmap.......")
            tmAP = CalcTopMap(trn_binary.numpy(), p_tst_binary.numpy(), trn_label.numpy(), p_tst_label.numpy(),
                              config["topK"])

            if tmAP > Best_tmAP:
                Best_tmAP = tmAP
            if mAP > Best_mAP:
                Best_mAP = mAP

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
            print("%s epoch:%d, bit:%d, dataset:%s, network:%s, Method:%s, MAP:%.3f, Best MAP: %.3f, t-MAP:%.3f, Best t-MAP: %.3f" % (
                    config["info"], epoch + 1, bit, config["dataset"], config["specific_type"], "HashNet", mAP, Best_mAP,
                    tmAP, Best_tmAP))


if __name__ == "__main__":
    config = get_config()
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print("Database:",config["dataset"], "Network:",config["specific_type"])

    # setup data sets
    if config["dataset"] == 'ImageNet':
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["MS-COCO"] == 'MS-COCO':
        config["dataset"] = 1000
        config["n_class"] = 80

    for bit in config["bit_list"]:
        print("Hash Bit:", bit)
        train_val(config, bit)