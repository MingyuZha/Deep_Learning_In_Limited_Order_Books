import numpy as np
import torch
from torch.autograd import Variable

from OrderBook import LimitOrderBook_magnitude


def _recall(model, x, y, category=4):
    correct = 0
    total = 0
    with torch.no_grad():
        x = Variable(x.float())
        if (torch.cuda.is_available()):
            x = x.cuda()
        output = model(x)
        for batch in range(len(x)):
            y_batch = y[batch].cpu().numpy()
            y_predict = output[batch]
            I = (y_batch == category)
            if (np.sum(I) > 0):
                y_recall = y_batch[I]
                # print (y_recall)
                if (category != 4):
                    probs = torch.cat((y_predict[:, :4], y_predict[:, 5:]), dim=1)
                    prediction = probs.data.max(1)[1]
                    prediction = prediction.cpu().numpy()
                    prediction[prediction >= 4] += 1
                    output_recall = prediction[I]
                    # print (output_recall)
                    correct += np.sum((y_recall == output_recall))
                    total += len(y_recall)
                else:
                    prediction = y_predict.max(1)[1]
                    prediction = prediction.cpu().numpy()
                    output_recall = prediction[I]
                    correct += np.sum((y_recall == output_recall))
                    total += len(y_recall)
    return {"correct": correct,
            "total": total}

def _precision(model, x, y, category):
    correct = 0
    total = 0
    with torch.no_grad():
        x = Variable(x.float())
        if (torch.cuda.is_available()):
            x = x.cuda()
        output = model(x)
        for batch in range(len(x)):
            y_batch = y[batch].cpu().numpy()
            if (category != 4):
                I = (y_batch != 4)
                y_batch = y_batch[I]
                y_predict = output[batch]

                probs = torch.cat((y_predict[:, :4], y_predict[:, 5:]), dim=1)
                prediction = probs.data.max(1)[1]
                prediction = prediction.cpu().numpy()
                prediction[prediction >= 4] += 1
                prediction = prediction[I]
                I = (prediction == category)
                if (np.sum(I) > 0):
                    y_precision = y_batch[I]
                    correct += np.sum(y_precision == prediction[I])
                    total += np.sum(I)
    if (total == 0): return {"acc": 0,
                            "total": total}
    return {"acc": float(correct * 100) / total,
            "total": total}

if __name__ == "__main__":
    dataset = LimitOrderBook_magnitude(root="/projects/sciteam/bahp/OneSecondData1000/", stock_name="GOOGL", train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)
    # iterator = iter(dataloader)
    # x, y = iterator.next()
    # for i in range(len(dataloader)-1):
    #     next_x, next_y = iterator.next()
    #     x = torch.cat((x, next_x), dim=0)
    #     y = torch.cat((y, next_y), dim=0)

    model = torch.load("./saved_model_GOOGL/GOOGL.model")
    model.cuda()
    model.eval()
    total_samples = 0
    # for category in range(9):
    #     result = _recall(model, x, y, category)
    #     print ("Total number of samples: %d, Recall accuracy: %.2f%%"%(result["total"], result["acc"]))
    #     total_samples += result["total"]
    # print ("Total samples: %d"%total_samples)

    for category in range(9):
        all_correct = 0
        all = 0
        for index, data in enumerate(dataloader):
            x, y = data
            result = _recall(model, x, y, category)
            all_correct += result["correct"]
            all += result["total"]
            # print ("Total number of samples: %d, Recall: %.2f%%"%(result["total"], result["acc"]))
            total_samples += result['total']
        print ("Category: %d, Total number of samples: %d, Recall: %.2f%%\n"%(category, all, float(all_correct*100)/all))
    print ("Total samples: %d"%total_samples)
