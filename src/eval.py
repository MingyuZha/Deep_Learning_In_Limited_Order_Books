import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.OrderBook import LimitOrderBook


def eval(model, root, stock_name, num_levels, input_size, sequence_len, batch_size):
    eval_dataset = LimitOrderBook(root=root, stock_name=stock_name, train=False, num_levels=num_levels,
                                  num_inputs=input_size, sequence_len=sequence_len)
    eval_dataLoader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_counter = 0
    correct_counter = 0
    running_loss = 0.
    with torch.no_grad():
        for index, data in enumerate(eval_dataLoader):
            x, y = data
            x_variable = Variable(x.float())
            if (torch.cuda.is_available()):
                x_variable = x_variable.cuda()
            output = model(x_variable)
            # print (y_predict.shape)
            batch_loss = Variable(torch.zeros(1))
            if (torch.cuda.is_available()):
                batch_loss = batch_loss.cuda()

            for time_step in range(sequence_len):
                ground_truth = Variable(torch.LongTensor(y[:, time_step].long()))
                if (torch.cuda.is_available()):
                    ground_truth = ground_truth.cuda()
                probs = torch.cat((output[:, time_step, 0:1], output[:, time_step, 2:]),
                                  dim=1)  ##[0:1] retains the column structure
                prediction = probs.data.max(1)[1]
                prediction_np = prediction.cpu().numpy()
                prediction_np[prediction_np == 1] = 2
                ground_truth_np = y[:, time_step].cpu().numpy()
                I = (ground_truth_np != 1)
                if (np.sum(I) > 0):
                    correct_counter += np.sum(ground_truth_np[I] == prediction_np[I])
                    total_counter += np.sum(I)
                batch_loss += F.nll_loss(output[:, time_step, :], ground_truth)

            batch_loss = batch_loss / float(sequence_len)
            running_loss += batch_loss.item()
    return {"acc": 100. * correct_counter / float(total_counter),
            "loss": running_loss / float(index + 1)}


if __name__ == "__main__":
    root = "../"
    stock_name = "AAPL"
    num_levels = 10
    input_size = 20
    sequence_len = 20
    batch_size = 100
    # print (len(dataset))
    model = torch.load("./saved_model_" + stock_name + "/" + stock_name + ".model", map_location='cpu')
    model.eval()
    result = eval(model)
    print("Evaluation accuracy: %.5f%%" % (result["acc"]))
    print("Evaluation loss: %.5f" % result["loss"])

