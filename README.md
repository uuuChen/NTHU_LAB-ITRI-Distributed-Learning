## 目前進度

#### DRD 資料集錯誤 (需刪除)
1. 194_right.jpeg
2. 29126_right.jpeg

#### 名詞說明
1. <新增> | 新增尚不存在的功能、註解、檔案...等
2. <更動> | 已有某功能且正常運作，更動其寫法、功能、註解...等
3. <除錯> | 已有某功能但出錯，將其更正

#### "uuuChen 08/15 01:20"
1. <更動> | "MNIST_dataSet.py" 裡 "get_data_and_labels": 原先並無支援
從 database 讀取 batch_size，更動為支援 batch_size
2. <新增> | "socket_/socket_.py": 參考 Version "Edward1997 08/14 17:30"，
將在　"MNIST_Server_train.py", "MNIST_Agent_train.py" 裡　server、
agent socket 的程式寫成獨立類別，讓使用時更為簡潔
3. <新增> | "socket_/socket_args.py": 此檔存放 socket 的一些 arguments，
與 "data_args" 的功能相近
4. <更動> | "MNIST_Server_train.py", "MNIST_Agent_train.py": 將程式碼以新
增的 "socket_.py" 改寫，另外將 "train_args"、"test_epoch()" 由 
"MNIST_Server_train.py" 改寫到 "MNIST_Agent_train.py"。另外原先在
"MNIST_Server_train.py" 以 "train_epoch()" 實作，改寫為 "iter_once()"，
如此能讓 server 彈性更高

#### "uuuChen 08/14 19:20"
1. <更動> | "train/" 裡 python 檔名稱
2. <新增> | "data_args" 裡加入 "DRD_TESTING_ARGS"，提供DRD部分資料測試

#### "Edward1997 08/14 17:30"
1. <新增> | "MNIST_Server_train.py", "MNIST_Agent_train.py"
使用 socket, pickle 在本地端進行 TCP 溝通的模型訓練
正確率與集中式相當

#### "Edward1997 08/14 14:40"
1. <新增> | "socket directory","socket_server.py","socket_client.py":
範例程式，使用 socket 的 TCP 傳輸資料，並利用 pickle 包裝物件
實現用網路傳輸Variable物件之功能

#### "uuuChen 08/13 17:20"
1. <新增> | "AlexNet.py" 裡 "Agent_AlexNet"、 "Server_AlexNet" : 將 AlexNet
根據 "classifier" 劃分為兩半，前半段在 agent、後半段在 server
2. <新增> | "dist_AlexNet_train.py" : 配合 "Agent_Alexnet" 、 
"Server_AlexNet" 進行訓練，訓練時集中式與分割式正確率相當接近

#### "uuuChen 08/13 16:35"
1. <除錯> | "dist_ECG_train.py" 裡 "train_epoch" : 原本分割式與集中式訓練時
正確率差異明顯，並不符合理論，後來發現是命名問題，因為 agent output 與
server input 都命名為 "features"，兩者的計算圖重疊而導致錯誤，後分別命名
為 "agent_output", "server_input"，訓練時集中式與分割式正確率相當接近

#### "Edward1997 08/13 14:00"
1. <新增> | "dis_MNIST_train.py":
單一程式，模型切割式訓練，訓練效果與模型未分割時相同。
接下來將繼續嘗試，若gradiant使用程式外傳來的是否依然可以成功。
2. <新增> | "LeNet.py" 中 "class Agent_LaNet","class Server_LeNet":
原先模型為五層(前2卷積層,後3全連接層)，分割為Server後3層，Agent前2層


#### "uuuChen 08/13 12:20"
1. <新增> | "MLP.py" 裡 "Agent_MLP"、 "Server_MLP" : 將四層的 MLP 從中間分割，
在 Agent 與 Server 端各配置兩層
2. <新增> | "dist_ECG_train.py" : 配合 "Agent_MLP" 、 "Server_MLP" 進行訓練
，雖然可以動但做法待驗證，原因是訓練速度與效果不如集中式，第二個 epoch 時集中式的正
確率可到 91 %、分割式才到 85% ，理論上兩者應該要一致  


#### "uuuChen 08/13 01:16"
1. <除錯、新增> | "data_processor.py" 裡 
"_make_sure_data_and_labels_in_database()" : 原本直接呼叫 
"coll_delete_all()" 而導致錯誤，因為忘記考慮 gridFS 的情況。目前調整為在 
"data_processor.py" 新增 "delete_coll_from_database()"，根據 use_gridFS
來決定如何刪除 collection 

#### "uuuChen 08/13 00:45"
1. <新增、更動、除錯> | "mongoDB_processor.py" 裡 "gridFS_coll_insert()" : 
新增「防止上傳不完整資料到資料庫」功能，否則在訓練時轉成 tensor 會出現 
"typeError"。另外更動其函式註解
2. <更動> | "DRD_train.py" 裡 "test_epoch()" : 原先會跑完所有 test data 並
取數值平均，但是太耗時，更動為隨機抽 "test_batch_size" 筆出來測試
3. <更動> | "data_processor.py" 裡 
"_make_sure_data_and_labels_in_database()" : 原本如果 database 某
collection 資料數與 local 資料數不同，便刪除整個database，
更動為只刪除該collection，像是 "train_data_labels" 或 "test_data_labels"
3. <除錯> | "AlexNet.py" 裡 "forward()" : F.log_softmax() 中傳入的 dim 由 
0 改為 1，計算才正確

#### "cleat README.md updatalist till 08/08"  | Edward1997 | 08/12

#### "vim mongoDB_processor.py"  | Edward1997 | 08/12
1. def gridFS_coll_insert
	label = int(labels[i]) 
	-->	label = labels[i]
2. def gridFS_coll_read_batch
	batch_labels.append(label)
	--->	label  = np.array(label)
        	batch_labels.append(label)

#### "vim DRD_train"  | Edward1997 | 08/12
1. def train_epoch
	data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size)
	-->	data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size, image_size= train_args.image_size)

#### "vim DRD_train"  | Edward1997 | 08/12
1. def test_epoch
	data, target = test_dataSet.get_data_and_labels(batch_size=train_args.batch_size)
	-->	data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size, image_size= train_args.image_size)

#### "add delete_all_database.py"  | Edward1997 | 08/12

#### "add centrl_dataset_import.py"  | Edward1997 | 08/12

#### "add Xray_dataset.py"  | Edward1997 | 08/12

#### "vim MNIST_train.py, DRD_train.py"  | Edward1997 | 08/12
1. def train_epoch
	datas_tratned_num += len(data)

#### "vim MNIST_dataSet.py" | Edward1997 | 08/08
1. 改為從 torchvision 下載本地端資料
2. 從資料庫讀資料改為一次全部讀取
3. 增加資料前處理(mongodb --> model 之間時處理)

#### "add LeNet.py" | Edward1997 | 08/08

#### "vim MNIST_train.py" | Edward1997 | 08/08
1. 參數調整: lr = 0.01 , batch_size = 128 , momentum = 0.9
2. 改用 LeNet模型
3. loss function 改用 cross_entropy()
4. 增加繪圖程式碼，但目前為註解狀態未啟用

#### "delet MNIST_data directory" , "add MNIST directory" | Edward1997 | 08/08
1. 更動本地存放之資料路徑與內容

#### "add readme.md" | uuuChen | 08/06
1. 增加 readme.md
2. 更動 "train.py" argparse 初始方式

#### "add train directory" | uuuChen | 08/06
1. 在 “train/” 創建對應 MNIST、DRD、ECG 的 python 檔，方便參數管理

#### "ECG train successfully" | uuuChen | 08/06
1. ECG 以 MLP train 成功，test dataSet 在第三個 epoch 的正確率大概 93%
2. 將 MNIST_train.py、DRD_train.py 的 model 放到 global，讓 train 跟 test
 時都使用同樣的 model。原先的做法在 test 時無法使用已被 train 更新的 model
3. MNIST 以 MLP 訓練，在 test 時的正確率也可到達 90% ，可嘗試將 model 以 
Lenet 替換避免已訓練資料顯示超過總資料上限
--------------------------------------------
