import numpy as np
import matplotlib.pyplot as plt
import os



def _show_different_stocks():
    stock_list = ["AAPL", "AMD", "CBS"]
    model_type = "NN"
    file_list = []
    if (model_type == "LSTM"):
        for stock in stock_list:
            f = open("../../outputs/"+stock+"_magnitude.OU")
            file_list.append(f)
    else:
        for stock in stock_list:
            f = open("../../outputs/"+stock+"_NN.OU")
            file_list.append(f)

    accuracy = {}
    loss = {}
    num_limit = 1000
    counter = 0

    for i in range(3):
        file = file_list[i]
        accuracy[stock_list[i]] = []
        loss[stock_list[i]] = []
        counter = 0
        for line in file:
            if (line.startswith("[") and counter < num_limit):
                counter += 1
                split_line = (line.strip("\n").split(":")[1]).split(",")
                tmp_acc = float(split_line[1][1:-1])
                tmp_loss = float(split_line[0][-7:])
                accuracy[stock_list[i]].append(tmp_acc)
                loss[stock_list[i]].append(tmp_loss)
        accuracy[stock_list[i]] = np.asarray(accuracy[stock_list[i]])
        loss[stock_list[i]] = np.asarray(loss[stock_list[i]])
    # print (loss)

    plt.figure()
    plt.subplot(121)
    plt.title("Training Accuracy")
    for i in range(3):
        plt.plot(accuracy[stock_list[i]], label=stock_list[i])
    plt.legend(loc="lower right")
    # for i in range(3):
    #     plt.plot()

    plt.subplot(122)
    plt.title("Training loss")
    for i in range(3):
        plt.plot(loss[stock_list[i]], label=stock_list[i])
    plt.legend(loc="upper right")
    plt.show()

    file.close()

def _show_one_stock():
    stock_name = "AAPL"
    job_id = "9710005"
    num_records_limit = 1000
    file = open("../../outputs/" + job_id + ".bw.out")
    training_loss = []
    eval_loss = []
    training_acc = []
    eval_acc = []
    num_records = 0
    for line in file:
        if (line.startswith("[") and num_records < num_records_limit):
            line = line.strip("\n").split(":")[1]
            data = line.split(",")
            training_loss.append(float(data[0][1:]))
            training_acc.append(float(data[1][1:-1]))
            eval_loss.append(float(data[2][1:]))
            eval_acc.append(float(data[3][1:-1]))
            num_records += 1
    training_loss = np.array(training_loss)
    training_acc = np.array(training_acc)
    eval_loss = np.asarray(eval_loss)
    eval_acc = np.asarray(eval_acc)
    plt.figure()
    plt.title(stock_name + "'s Training and Evaluation loss")
    plt.plot(training_loss, label="Training")
    plt.plot(eval_loss, label="Evaluation")
    plt.legend(loc="upper right")
    plt.show()
    file.close()


def _show_AMD(type):
    if (type == "different sequence length"):
        sequence_len = [20, 30 ,40, 50]
        training_loss = {}
        for length in sequence_len:
            training_loss[length] = []
            file_name = "../../outputs/AMD/sl=" + str(length) + ".evaluation_loss.txt"
            file = open(file_name)
            for line in file:
                training_loss[length].append(float(line.strip("\n")[:-1]))
            training_loss[length] = np.asarray(training_loss[length])
        plt.figure()
        for length in sequence_len:
            plt.plot(training_loss[length], label="sequence len:"+str(length))
        plt.legend(loc="upper right")
        plt.show()
    elif (type == "training loss versus evaluation loss"):
        training_loss = []
        eval_loss = []
        training_acc = []
        eval_acc = []
        with open("../../outputs/AMD/sl=20.training_loss.txt") as file:
            for line in file:
                training_loss.append(float(line.strip("\n")))
        training_loss = np.asarray(training_loss)
        with open("../../outputs/AMD/sl=20.evaluation_loss.txt") as file:
            for line in file:
                eval_loss.append(float(line.strip("\n")))
        eval_loss = np.asarray(eval_loss)
        with open("../../outputs/AMD/sl=20.training_acc.txt") as file:
            for line in file:
                training_acc.append(float(line.strip("\n")[:-1]))
        training_acc = np.asarray(training_acc)
        with open("../../outputs/AMD/sl=20.evaluation_acc.txt") as file:
            for line in file:
                eval_acc.append(float(line.strip("\n")[:-1]))
        eval_acc = np.asarray(eval_acc)
        plt.figure()
        plt.subplot(121)
        plt.title("Loss")
        plt.plot(training_loss, label="Training loss")
        plt.plot(eval_loss, label="Evaluation loss")
        plt.legend(loc="upper right")
        plt.subplot(122)
        plt.title("Accuracy")
        plt.plot(training_acc, label="Training accuracy")
        plt.plot(eval_acc, label="Evaluation accuracy")
        plt.legend(loc="lower right")
        plt.show()



if __name__ == "__main__":
    _show_AMD("different sequence length")