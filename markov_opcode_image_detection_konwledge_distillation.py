# -*- coding: utf-8 -*-
"""
@Author: Liangwei Yao
@Date: 2025/4/13 14:43
@FileName: get_permission.py
@Description: 
"""
import ast
import copy
import itertools
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from MobileNetV3 import mobilenet_v3_large, mobilenet_v3_small
import time
from datetime import datetime
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from torch.utils.data import DataLoader


# 模型训练函数
def train_model(teacher_model, student_model, train_loader, valid_loader=None, epochs=20, filename="best_model.pth"):
    # 初始化变量
    since = time.time()
    best_training_acc = 0
    best_val_acc = 0
    train_acc_history = []
    val_acc_history = []
    train_losses_history = []
    val_losses_history = []
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(student_model.state_dict())
    T = 5.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 100)
        student_model.train()
        teacher_model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            with torch.set_grad_enabled(True):
                student_output, datalist = student_model(inputs)
                _, preds = torch.max(student_output, 1)
                hard_loss = criterion(student_output, labels)
                soft_loss = F.kl_div(F.log_softmax(student_output / T, dim=1),
                                     F.softmax(teacher_output / T, dim=1),
                                     reduction='batchmean') * T * T
                alpha = 0.5
                loss = hard_loss * (1 - alpha) + soft_loss * alpha
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_acc_history.append(epoch_acc)
        train_losses_history.append(epoch_loss)
        time_elapsed = time.time() - since
        print('Epoch {} Training Time elapsed {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
        print('Epoch {} Training Loss: {:.4f} Training Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
        if epoch_acc > best_training_acc and valid_loader is None:
            best_training_acc = epoch_acc
            best_model_wts = copy.deepcopy(student_model.state_dict())
            state = {
                'state_dict': student_model.state_dict(),
                'best_training_acc': best_training_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, filename)
        if valid_loader is not None:
            val_loss = 0
            valid_correct = 0
            student_model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    val_outputs, datalist = student_model(inputs)
                    loss = criterion(val_outputs, labels)
                    _, preds = torch.max(val_outputs, 1)
                    valid_correct += (preds == labels).sum().item()
                    val_loss += loss.item() * inputs.size(0)
                val_loss = val_loss / len(valid_loader.dataset)
                val_acc = valid_correct / len(valid_loader.dataset)
            val_acc_history.append(val_acc)
            val_losses_history.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(student_model.state_dict())
                state = {
                    'state_dict': student_model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete: %.04fs' % time_elapsed)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Training Acc: {:4f}'.format(best_training_acc))
    if valid_loader is not None:
        print('Best val Acc: {:4f}'.format(best_val_acc))
    student_model.load_state_dict(best_model_wts)
    current_time = datetime.now()
    file = open(os.path.join(picture, "data.txt"), "a")
    model_name = filename.split("/")[2].split(".")[0]
    file.write(model_name + "-write time:" + str(current_time) + "\n")
    file.write("val_acc_history:" + f"{val_acc_history}" + "\n" + "train_acc_history:" + f"{train_acc_history}" + "\n"
               + "val_losses_history:" + f"{val_losses_history}" + "\n" + "train_losses_history:" + f"{train_losses_history}" + "\n")
    file.close()
    return student_model, train_acc_history, val_acc_history, train_losses_history, val_losses_history, LRs


def test_model(model, test_loader, filename):
    running_corrects = 0
    test_preds = []
    test_labels = []
    test_datalist = []
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, batch_datalist = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_datalist.append(batch_datalist.cpu().numpy())
    time_elapsed = time.time() - since
    print('test complete: %.04f s' % time_elapsed)
    print('test complete {:.4f}s'.format(time_elapsed))
    print('test complete {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    datalist = np.concatenate(test_datalist, axis=0)
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)

    manual_accuracy = running_corrects / len(test_labels)
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=['benign', 'malware'])
    cm = confusion_matrix(test_labels, test_preds)
    print('Test Acc: {:.4f}'.format(manual_accuracy))
    print(f"Test Acc: {accuracy:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    precision = precision_score(test_labels, test_preds, average='macro')
    recall = recall_score(test_labels, test_preds, average='macro')
    f1 = f1_score(test_labels, test_preds, average='macro')
    print('macro Test Precision: {:.2f}%'.format(100. * precision))
    print('macro Test Recall: {:.2f}%'.format(100. * recall))
    print('macro Test F1: {:.2f}%'.format(100. * f1))

    precision = precision_score(test_labels, test_preds, average='weighted')
    recall = recall_score(test_labels, test_preds, average='weighted')
    f1 = f1_score(test_labels, test_preds, average='weighted')
    print('weighted Test Precision: {:.2f}%'.format(100. * precision))
    print('weighted Test Recall: {:.2f}%'.format(100. * recall))
    print('weighted Test F1: {:.2f}%'.format(100. * f1))

    precision = precision_score(test_labels, test_preds, average='micro')
    recall = recall_score(test_labels, test_preds, average='micro')
    f1 = f1_score(test_labels, test_preds, average='micro')
    print('micro Test Precision: {:.2f}%'.format(100. * precision))
    print('micro Test Recall: {:.2f}%'.format(100. * recall))
    print('micro Test F1: {:.2f}%'.format(100. * f1))

    return accuracy, report, cm, datalist, test_labels


apk_dir = os.path.abspath('../../at2/datasets')
# 加载数据集
transform = transforms.Compose([  # transforms.Resize(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=os.path.join(apk_dir, "processedapk2020"), transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

student_model = mobilenet_v3_small(reduced_tail=True).to(device)
criterion = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss()
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0,
                                                       T_max=50)
filename = './models/resnet18_mobile3_small_r_ACA_adamW_CE_8_1_1_new.pth'
file_dir = os.path.dirname(filename)
print(file_dir)
picture = './picture/resnet18_mobile3_small_r_ACA_adamW_CE2020apk'
if not os.path.exists(picture):
    os.makedirs(picture)

checkpoint = torch.load(filename, map_location=device)
student_model.load_state_dict(checkpoint['state_dict'])

# 分类器评估
accuracy, report, cm, datalist, test_labels = test_model(student_model, test_loader, filename)
