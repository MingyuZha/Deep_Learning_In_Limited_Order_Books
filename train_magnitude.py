import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from OrderBook import LimitOrderBook_magnitude
from lstm import LSTM_model

def eval():
    model.eval()
    total_counter = 0
    correct_counter = 0
    running_loss = 0.
    counter = 0
    len_dataset = eval_dataset.__len__()
    with torch.no_grad():
        for i, data in enumerate(eval_dataLoader):
            x, y = data
            counter += 1
            x_variable = Variable(x.float())

            if (torch.cuda.is_available()):
                x_variable = x_variable.cuda()
            output = model(x_variable)
            # print (y_predict.shape)
            batch_loss = Variable(torch.zeros(1))
            if (torch.cuda.is_available()):
                batch_loss = batch_loss.cuda()

            for time_step in range(sequence_len):
                ground_truth = Variable(y[:, time_step].long())
                if (torch.cuda.is_available()):
                    ground_truth = ground_truth.cuda()
                probs = torch.cat((output[:, time_step, 0:4], output[:, time_step, 5:]),
                                  dim=1)  ##[0:1] retains the column structure
                prediction = probs.data.max(1)[1]
                prediction_np = prediction.cpu().numpy()
                prediction_np[prediction_np >=4] += 1
                ground_truth_np = y[:, time_step].cpu().numpy()
                I = (ground_truth_np != 4)
                if (np.sum(I) > 0):
                    correct_counter += np.sum(ground_truth_np[I] == prediction_np[I])
                    total_counter += np.sum(I)
                batch_loss += F.nll_loss(output[:, time_step, :], ground_truth)

            batch_loss = batch_loss / float(sequence_len)
            running_loss += batch_loss.item()
    model.train()
    return {"acc": 100. * correct_counter / float(total_counter),
            "loss": running_loss / float(counter)}


root = "../"
stock_name = "AAPL"


model_dir = "./saved_model_" + stock_name + "/"
output_dir = "./outputFiles/" + stock_name + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

###################################################################
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.001
sequence_len = 30
input_size = 20
hidden_size = 200  ##200
fc_size = 50  ##50
num_levels = 10
num_layers = 3  ##3
scheduler_step_size = 10
scheduler_gamma = .5
num_classes = 9

print ("------------------------------------------------------------")
print ("Batch size: %d\n"
       "Learning rate: %.3f\n"
       "Sequence Length: %d\n"
       "Number of hidden units in LSTM: %d\n"
       "Number of hidden units in FC layer: %d\n"
       "Number of layers in LSTM: %d\n"
       "Number of classes: %d"
       %(BATCH_SIZE, LR, sequence_len, hidden_size, fc_size, num_layers, num_classes))
print ("------------------------------------------------------------")
###################################################################




Dataset = LimitOrderBook_magnitude(root=root, stock_name=stock_name, train=True, num_levels=num_levels, num_inputs=input_size,
                         sequence_len=sequence_len)
DataLoader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

eval_dataset = LimitOrderBook_magnitude(root=root, stock_name=stock_name, train=False, num_levels=num_levels,
                                  num_inputs=input_size, sequence_len=sequence_len)
eval_dataLoader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


model = LSTM_model(num_layers=num_layers,
                   input_size=input_size,
                   hidden_size=hidden_size,
                   fc_size=fc_size,
                   output_size=num_classes)

if (torch.cuda.is_available()): model.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
# loss_fn = nn.BCEWithLogitsLoss()
## NLL_Loss和CrossEntropy Loss相同，区别是后者为我们做了softmax操作
# loss_fn = nn.nll_loss() ##NLL Loss的输入是一个对数概率向量和一个目标标签，他不会为我们计算对数概率，适合网络的最后一层是log_softmax

# train_loss_chart = []
for epoch in range(EPOCHS):
    model.train()
    scheduler.step()
    running_loss = 0.
    correct_counter = 0
    total_counter = 0

    for index, data in enumerate(DataLoader):
        optimizer.zero_grad()
        ## x: [batch_size, sequence_len, 20]
        ## y: [batch_size, sequence_len, 1]
        x, y = data
        ## y_hat: [batch_size, sequence_len, 1]
        x_variable = Variable(x.float())
        if (torch.cuda.is_available()):
            x_variable = x_variable.cuda()
        output = model(x_variable)
        # print (y_predict.shape)
        batch_loss = Variable(torch.zeros(1))
        if (torch.cuda.is_available()):
            batch_loss = batch_loss.cuda()

        # avg_error = 0.
        for time_step in range(sequence_len):
            ground_truth = Variable(torch.LongTensor(y[:, time_step].long()))
            if (torch.cuda.is_available()):
                ground_truth = ground_truth.cuda()

            #Calculate classification accuracy
            probs = torch.cat((output[:, time_step, 0:4], output[:, time_step, 5:]),
                              dim=1)  ##[0:1] retains the column structure
            prediction = probs.data.max(1)[1]
            prediction_np = prediction.cpu().numpy()
            prediction_np[prediction_np >= 4] += 1
            ground_truth_np = y[:, time_step].cpu().numpy()
            I = (ground_truth_np != 4)
            if (np.sum(I) > 0):
                correct_counter += np.sum(ground_truth_np[I] == prediction_np[I])
                total_counter += np.sum(I)
            batch_loss += F.nll_loss(output[:, time_step, :], ground_truth)

        batch_loss = batch_loss / float(sequence_len)
        running_loss += batch_loss.item()
        batch_loss.backward()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if ('step' in state and state['step'] >= 1024):
                    state['step'] = 1000
        optimizer.step()

        if ((index + 1) % 10 == 0):
            train_acc = correct_counter * 100. / total_counter
            with open(output_dir + "sl=" + str(sequence_len) + ".training_loss.txt", "a") as f:
                f.write("%.5f\n" % (running_loss / 10.))
            with open(output_dir + "sl=" + str(sequence_len) + ".training_acc.txt", "a") as f:
                f.write("%.5f%%\n" % (train_acc))
            ##Evaluation phase
            eval_result = eval()
            eval_acc = eval_result["acc"]
            eval_loss = eval_result["loss"]
            with open(output_dir + "sl=" + str(sequence_len) + ".evaluation_loss.txt", "a") as f:
                f.write("%.5f\n" % eval_loss)
            with open(output_dir + "sl=" + str(sequence_len) + ".evaluation_acc.txt", "a") as f:
                f.write("%.5f\n" % eval_acc)
            print("[%d, %d]: %.5f, %.5f%%, %.5f, %.5f%%" % (index + 1, len(DataLoader), running_loss / 10.,
                                                               100. * correct_counter / total_counter, eval_loss,
                                                               eval_acc))
            running_loss = 0.
            correct_counter = 0
            total_counter = 0
            torch.save(model, model_dir + stock_name + ".model")



