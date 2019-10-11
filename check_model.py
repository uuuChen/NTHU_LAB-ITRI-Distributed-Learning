from train.switch import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np

def test():
    model.eval()
    _, dataSet = switch.get_dataSet(shuffle=True)
    batch_size = 10

    data_nums = dataSet.get_usage_data_nums()

    targets = []
    preds = []
    trained_data_num = 0
    total_loss = 0
    correct = 0
    batches = (data_nums - 1) // batch_size + 1
    for batch_idx in range(1, batches + 1):
        data, target = dataSet.get_data_and_labels(batch_size=batch_size)

        data, target = Variable(data).float(), Variable(target).long()

        if train_args.cuda:
            data, target = data.cuda(), target.cuda()


        output = model(data)
        loss = F.cross_entropy(output, target)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        total_loss += loss.item()

        targets.extend(target.data.cpu())
        preds.extend(pred.data.cpu())

        total_loss /= batches

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, correct, data_nums, 100. * correct / data_nums))

    plot_confusion_matrix(target=targets, pred=preds,
                           classes=np.array(list(switch.data_args[1]['class_id'].keys())),
                           data_name=data_name)


def plot_confusion_matrix(target, pred, classes, data_name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
    cm = confusion_matrix(target, pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if normalize:
        plt.savefig(save_path + data_name + "_central_confusion_matrix(normalize).png", dpi=300, format="png")
    else:
        plt.savefig(save_path + data_name + "_central_confusion_matrix.png", dpi=300, format="png")

data_name = 'MNIST'
switch = Switch(data_name=data_name)
train_args = switch.get_train_args()
train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()
save_path = "record/10_11/"+data_name+"/"+data_name+"_model.pkl"
model = torch.load(save_path)
test()