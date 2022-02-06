import time
import torch
from torch import nn
from torch import optim
import torch.utils.data as data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from model.PSPNet import PSPNet
from data_loader import make_data_path_list, DataTransform, VOCDataset


def train_model(net, data_loaders_dict, optimizer, num_epochs):
    # check if GPU is available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')

    net.to(device)

    # If the network is somewhat fixed, make it faster
    cudnn.benchmark = True

    # number of image
    num_train_image = len(data_loaders_dict["train"].dataset)
    num_val_image = len(data_loaders_dict["val"].dataset)

    # setting up iteration counter
    iteration = 1

    # multiple minibatch
    batch_multiplier = 4

    # loop of epoch
    for epoch in range(num_epochs):

        # save the start time
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('--------------')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('--------------')

        # training and validation loop for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                # scheduler.step()
                optimizer.zero_grad()
                print('(train)')
            else:
                net.eval()
                print('-------------')
                print('(val)')

            # Loop to extract each minibatch from the data loader
            count = 0
            for images, anno_class_images in data_loaders_dict[phase]:
                if images.size()[0] == 1:
                    continue

                images = images.to(device)
                anno_class_images = anno_class_images.to(device)
                anno_class_images = anno_class_images != 0

                # Update parameters in multiple minibatches
                if phase == 'train' and count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                # forward propagation calculation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)
                    loss = F.cross_entropy(outputs, anno_class_images.long(), reduction="mean") / batch_multiplier

                    # Back-propagation during training
                    if phase == 'train':
                        loss.backward()
                        count -= 1

                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(
                                f'iteration {iteration} || Loss: {loss.item() * batch_multiplier} || '
                                f'10iter: {duration:.4f} sec.')
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    # Time of validation
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier
        # Loss and percentage of correct answers for each phase of epoch
        t_epoch_finish = time.time()
        print('--------------')
        print(
            f'epoch {epoch + 1} || Epoch_train_Loss: {epoch_train_loss / num_train_image} || Epoch_val_Loss: '
            f'{epoch_val_loss / num_val_image}')

        print(f'timer: {t_epoch_finish - t_epoch_start}')

        # Save the last network
        if (epoch + 1) % 10 == 0:
            torch.save(net.to('cpu').state_dict(), 'weights/pspnet50_2_' + str(epoch + 1) + '.pth')
            net.to(device)


if __name__ == '__main__':
    # Initial values
    text_path = "D:\\LearningData/supervisely_person_clean_2667_img/"
    data_path = "D:\\LearningData/supervisely_person_clean_2667_img/"
    batch_size = 8
    class_num = 21
    num_epochs = 50

    # make file path list
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_data_path_list(text_path=text_path,
                                                                                       data_path=data_path)

    # make Dataset
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                               transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                             transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # combine into a dictionary type variable
    data_loaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # replace convolutional layer for classification with the one with 21 outputs
    net = PSPNet(class_num=21)

    # initialize the replacement convolutional layer.
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


    net.decoder.classification.apply(weights_init)

    # setting up the learning late
    optimizer = optim.RAdam(net.parameters(), lr=0.001)

    # Do training and validation
    print(net)
    train_model(net, data_loaders_dict, optimizer, num_epochs=num_epochs)
