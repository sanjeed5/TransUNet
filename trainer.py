import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Splitting to validation dataset
    val_split = 0.2
    dataset_size = len(db_train)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(db_train,
                                               [train_size, val_size])


    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    print(f"\nLength of dataset: {len(db_train)}, train: {train_size}, val: {val_size}")

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    min_valid_loss = np.inf
    min_valid_loss_epoch = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        train_loss_avg = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            print("outputs.shape: ", outputs.shape, ", label_batch[:].long().shape: ", label_batch[:].long().shape)
            loss_bce = bce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            train_loss_avg += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_bce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_bce.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Validation loop
        val_loss_avg = 0
        if (epoch_num + 1) %  args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(val_loader):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    val_loss_bce = bce_loss(outputs, label_batch[:].long())
                    val_loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    val_loss = 0.5 * val_loss_bce + 0.5 * val_loss_dice
                    val_loss_avg += val_loss.item()
                
                train_loss_avg = train_loss_avg/len(trainloader)
                val_loss_avg = val_loss_avg/len(val_loader)
                logging.info('epoch %d : avg_train_loss : %f, avg_val_loss: %f' % (epoch_num, train_loss_avg, val_loss_avg))

                writer.add_scalar('info/val_loss', val_loss_avg, epoch_num)
                
                writer.add_scalars(f'loss', {
                            'train_loss': train_loss_avg,
                            'val_loss': val_loss_avg,
                        }, epoch_num)

                if min_valid_loss > val_loss_avg:
                    min_valid_loss = val_loss_avg
                    min_valid_loss_epoch = epoch_num

                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

        
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    
    logging.info(f"\nLowest validation loss: {min_valid_loss:.3f} at epoch number: {min_valid_loss_epoch}")

    writer.close()
    return "Training Finished!"