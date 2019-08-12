## 目前進度

#### DRD 資料集錯誤 (需刪除)
1. 194_right.jpeg
2. 29126_right.jpeg

#### 名詞說明
1. <新增> | 新增尚不存在的功能、註解、檔案...等
2. <更動> | 已有某功能且正常運作，更動其寫法、功能、註解...等
3. <除錯> | 已有某功能但出錯，將其更正

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
