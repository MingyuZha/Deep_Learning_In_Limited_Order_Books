# Deep Learning Model for High-frequency Financial Data
## Introduction
Nowadays, deep learning has shown its power in tramendous areas, like computer vision, natural languange processing, etc. However, this project focuses on applying deep learning methods in high-frequency financial data --- Limited Order Books, to predict the next mid price move direction and magnitude. 
## Data
The data we use for this project is fixed time interval data, it can be requested on https://lobsterdata.com/. 
The data file contains three keys:
* `X_date`: records the limited order book
* `Y_date`: records the next best ask and bid price in Fixed Time Scale
* `T_date`: 
  * a. Time stamp(Start from the first row and keep adding the time interval);
  * b. transaction time just after the current time stamp;
  * c. current accumulated transaction number( Corresponding to the row numbers in the original limited order book)
 
## Preprocessing
The raw limited order book state has 40 dimensions. The format is like this: 

| Col 1        | Col 2           | Col3  | Col4 | Col5 | Col6 | ...
|:-------------:|:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Best ask price | Best ask size | Best bid price | Best bid size | 2nd ask price | 2nd ask size | ...


However, the spread of ask and bid prices changes through time, and the price magnitude barely have meaningful information we need. Therefore, we have to:
1. Binning the price spread. Here we set 20 bins ( 10 for ask price, 10 for bid price )
2. Put the corresponding sizes into the bins.

## Model
Since the limited order book state has inherent time dependency, it would be beneficial to utilize a sequence of state when we do the prediction job. Hence, we decide to use **LSTM** model for this project.
PyTorch has a built-in module so we do not have to write the LSTM layer on our own. The model structure is shown below:
``` python
class LSTM_model(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, fc_size, output_size):
        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_layers=num_layers,
                            input_size = self.input_size,
                            hidden_size = self.hidden_size,
                            batch_first = True,
                            dropout =  0.2)
        self.fc = nn.Linear(hidden_size, fc_size)
        self.decoder = nn.Linear(fc_size, output_size)

    def forward(self, order_book_data):
        """
        order_book_data: [batch_size, sequence_len, dims]
        """
        lstm_output, _ = self.lstm(order_book_data, None)
        output = F.relu(self.fc(lstm_output))
        output = self.decoder(output)
        return F.log_softmax(output, dim=2)
```
When we return the output, we do a `log_softmax` operation, since we need the output to be the probability of each class the input data can be, and the `log` operation is for the later on loss calculation's convenience. 

## Training
* **Optimizer**: `Adam`
* **Loss function**: `nll_loss`
* **Learning rate**: 0.001
* **Truncation Length**: 20
* **Batch size**: 256
* **Hidden units**: 200

## Results
After some experiments, we found that the performance of our model varies from different stocks. We used traditional **feed-forward neural network** as baseline. I put a table below to show the Evaluation accuracy on different stocks.

|        | AAPL | AMD  | CBS |
|:------:|:-----:|:-----:|:-----:|
| LSTM | 62.23% | 72.84% | 47.82% |
| Neural Network | 58.52% | 67.29% | 45.13% |


We also printed the training and evaluation loss:






