# distributed_learning
--------------------------------------------
## 初始 origin 參數 (只需執行一次）
1. git remote add origin https://github.com/uuuChen/distributed_learning  
--------------------------------------------
## 上傳
1. git add . <br>
2. git commit -m "commit message" <br>
3. git push <br>

--------------------------------------------
## 更新
1. git pull <br>

## 強制更新
1. git reset --hard <br>
2. git pull <br>

--------------------------------------------
## 目前進度

####  "add readme.md" | uuuChen | 08/06
1. 增加 readme.md <br>
2. 更動 "train.py" argparse 初始方式 <br>

####  "add train directory" | uuuChen | 08/06
1. 在 “train/” 創建對應 MNIST、DRD、ECG 的 python 檔，方便參數管理<br>

#### "ECG train successfully" | uuuChen | 08/06
1. ECG 以 MLP train 成功，test dataSet 在第三個 epoch 的正確率大概 93%<br>
2. 將 MNIST_train.py、DRD_train.py 的 model 放到 global，讓 train 跟
 test 時都使用同樣的 model。原先的做法在 test 時無法使用已被 train 更新的 model<br>
3. MNIST 以 MLP 訓練，在 test 時的正確率也可到達 90% ，可嘗試將 model 以 Lenet 替換
--------------------------------------------
