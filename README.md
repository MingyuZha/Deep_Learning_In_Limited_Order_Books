# Deep Learning Model for High-frequency Financial Data
## Introduction
Nowadays, deep learning has shown its power in tramendous areas, like computer vision, natural languange processing, etc. However, this project focuses on applying deep learning methods in high-frequency financial data --- Limited Order Books, to predict the next mid price move direction and magnitude. </br>
## Data
The data we use for this project is fixed time interval data, it can be requested on https://lobsterdata.com/?gclid=CjwKCAjw96fkBRA2EiwAKZjFTTorZkjxj5z8Ti9XG8jsRYpFnNXaDEVQDQqaUUTfqvok7elOQRmw0hoCMx4QAvD_BwE. </br>
The data file contains three keys:
* X_date: records the limited order book
* Y_date: records the next best ask and bid price in Fixed Time Scale
* T_date: 
  * a. Time stamp(Start from the first row and keep adding the time interval);
  * b. transaction time just after the current time stamp;
  * c. current accumulated transaction number( Corresponding to the row numbers in the original limited order book)

