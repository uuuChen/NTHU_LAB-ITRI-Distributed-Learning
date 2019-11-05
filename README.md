## 目前進度

#### DRD 資料集錯誤 (需刪除)
1. 194_right.jpeg
2. 29126_right.jpeg

#### 名詞說明
1. <新增> | 新增尚不存在的功能、註解、檔案...等
2. <更動> | 已有某功能且正常運作，更動其寫法、功能、註解...等
3. <除錯> | 已有某功能但出錯，將其更正
#### "Edward1997 11/04 18:20"
1. <除錯> | "distributed_train" 中 func plot_contusion_matrix
    * label error

#### "uuuChen 11/05 14:20"
1. <新增> | model/AlexNet.py
    1. Server_AlexNet_2, Agent_AlexNet_2:  新增不同切割層的 AlexNet

2. <更動，新增> | distributed_train/switch.py
    1. 加入一組新的 data_name "MC_2" ，他會使用 "Server_AlexNet_2", "Agent_AlexNet_2" 而非 "Server_AlexNet", "Agent_AlexNet"

#### "uuuChen 10/30 00:20"
1. <更動> | distibuted_server_train.py, distibuted_agent_train.py
    1. 實作改為 for 迴圈，另外將 agent sleep 時間從 50 更動為 100
2. <更動> | train_args.py
    1. 將 train_agrs 中 OCT 的 epoch 從 80 改為 50

#### "Edward1997 10/29 13:50"
1. <更動> | central_train.py
    1. 每 5 epochs 紀錄一次模型
    2. 每次執行前詢問是否要從已有模型繼續，追問模型時間 + 起始 epoch
    3. 繪製 acc, loss 圖不再藉由陣列紀錄，改為讀取 record 檔中數值
2. <更動> | server.py, agent.py
    1. 每 5 epochs 紀錄一次模型
    2. 每次執行前詢問是否要從已有模型繼續，追問模型路徑 + 起始 epoch
    3. 繪製 acc, loss 圖不再藉由陣列紀錄，改為讀取 record 檔中數值
    4. 傳送 is_training_done 改為 cur_epoch，再讓 agent 自行判斷是否結束 ( 因為 train_args 已包含 total epoch )
    5. train_args 增加 save_path，讓 server, agent 存放模型之路徑相同，讓中斷回復更易實現
    6. 紀錄 snapshot 起訖時間
3. <更動> | distibuted_server_train.py, distibuted_agent_train.py
    1. 連續跑 4 個 dataset，不中斷

#### "uuuChen 10/28 01:15"
1. <更動> | 將 MD 正名為 MC

#### "Edward1997 10/24 17:00"
1. <更動> | Server.py、Agent1.py、central_dataset_import.py 正名
    1. Server.py -> distributed_server_train.py
    2. Agent1.py -> distributed_agent_train.py
    3. central_dataset_import.py -> dataset_import.py

#### "Edward1997 10/21 12:00"
1. <更動> | central_train.py、Server.py、Agent1.py、central_dataset_import.py、 delete_all_database.py
    使其可用 terminal 運行
2. <更動> | central_train.py、Server.py、Agent1.py
    使其可由 terminal 讀取參數
    1. central_train.py data_name
    2. distributed_server_train.py data_name agent_nums
    3. distributed_agent_train.py agent_num server_IP

#### "Edward1997 10/18 16:20"
1. <更動> | central_train.py, server.py
    使其準確率 ( accuracy ) 精確至小數點

#### "Edward1997 10/18 14:50"
1. <新增> | server.py
    * 效能紀錄完善
2. <除錯> | server.py
    * 解決 confusion matrix 超界問題


#### "Edward1997 10/17 18:20"
1. <除錯> | import 問題
    * 若在 cmd 運行，需增加
    ```
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    ```
    並註解掉 `os.chdir('../../')`
    * 若在 pycharm 運行，則反之
    增加 `os.chdir('../../')`
    註解掉 `sys.path.insert(0, os.getcwd())`

#### "uuuChen 10/15 13:30"

1. <新增> | "data/MD/MD_preprocessing.py": 對 MD 資料進行欲處理

2. <更動> | "central_dataset_import.py", "delete_all_database.py": 以
"switch" 改寫，並移動至 "dataSet/"

3. <更動> | 整理程式碼，刪除 "DRD"、"Xray" 相關程式碼，更動檔案有
"data_args.py"、"train_args.py"、"switch.py"，並刪除 "DRD_dataSet.py"、
"Xray_dataSet.py"。另外在 "socket/" 只保留 "socket_.py"，其他的檔案則刪除

4. <新增> | "data_processor.py" init 中判斷有無 "/data_nums" 資料夾，如果
沒有的話就新增一個

#### "Edward1997 10/11 20:50"
1. <除錯> | "distributed_train" 、"central_train.py" 中 func plot_contusion_matrix
    1. 縱座標超格問題

#### "Edward1997 10/08 17:15"
1. <新增> | "distributed_train" 紀錄功能
    1. plot_confusion_matrix，紀錄 confusion_matrix
2. <更動> | "dataSet.py"、"data_args.py"
    "dataSet.py" : 不必新增 self.class_id 參數
    "data_args.py" : 作為替代，需在各 _TEST_ARGS 中增加 'class_id' 參數


#### "Edward1997 10/05 22:30"
1. <新增> | "distributed_train" 紀錄功能
    1. record_time : txt檔，紀錄訓練開始與結束時間，
    紀錄 train loss、train accuracy、test loss、test accuracy，每 epoch 為單位
    2. plot_acc_loss : 圖檔train、test loss 比較圖、train、test accuracy 比較圖


#### "Edward1997 10/05 11:55"
1. <新增> | "central_train" 函數 plot_confusion_matrix，紀錄 confusion_matrix
須在各 dataSet.py 中新增 self.class_id 參數

#### "uuuChen 10/1 22:10"
1. <新增> | "OCT_dataSet.py", 以及相關 code 使其可進行分散式訓練

#### "uuuChen 10/1 19:00"
1. <新增> | "train/DRD": 以數據集增強集中式訓練 DRD，並且沒有經過 dataBase
2. <新增> | "data_processor.py" 裡 "_up_sampling": 根據 local 的 csv 檔，
決定要如何數據增強。benchmark 為 n 表示以第 n 高的 label 為基準，讓比基準少的
label 進行數據增強

#### "Edward1997 09/24 12:50"
1. <新增> | "Xray_augmentation.py" 資料增強功能
    實現映射、亮暗
1. <更動> | "Xray_preprocessing.py" 
    根據增強資料，csv 中一筆 label 可以對應到多筆 img
    
#### "Edward1997 09/18 12:50"
1. <新增> | "central_train.py" early stop 功能

#### "Edward1997 09/18 12:50"
1. <新增> | "CatDog_dataSet" 以及相關貓狗訓練的程式碼

#### "Edward1997 09/14 9:55"
1. <更動> | "central_train.py"
    紀錄效能方式調整
    1. txt檔，紀錄訓練開始與結束時間，紀錄 train loss、train accuracy、test loss、tes accuracy 每 epoch 為單位
    2. 圖檔train、test loss 比較圖、train、test accuracy 比較圖
2. <更動> | "VGGNet.py"
    增加Dropout layers
#### "uuuChen 09/11 13:00"
1. <更動> | "train_args" 裡 "DRD": 將 "test_batch_size" 從 500 改為 100
2. <更動> | 'server.py': 與  "Edward1997 09/11 9:20" 的更動相同

#### "Edward1997 09/11 9:20"
1. <除錯> | "central_train.py"
    test_loss += loss --> test_loss += loss.item()
    減緩記憶體不足問題
    在更動前，Xray batch_size = 1, image_size = (128, 128) 也訓練不起來

#### "Edward1997 09/10 17:20"
1. <新增> | "central_train.py", "distributed_train/server.py" 中 save_acc 功能
    儲存 train loss 以及 test loss, accuracy
2. <更動> | "distributed_train/server.py" loss function
    null_loss -> cross_entropy

#### "Edward1997 09/07 15:20"
1. <更動> | 正名 : 
    1. local_central_train 資料夾 --> central_train 資料夾
    2. local_central_train.py --> central_train.py
    3. central.py --> switch.py
    4. local_split_train 資料夾 --> central_split_train 資料夾
    5. local_split_train.py --> central_split_train.py
2. <更動> | "model/*.py"、"central_train.py"、"central_split_train.py": 
    1. central_train.py 和 central_split_train.py 中使用 cross_entropy() 取代 nll_loss() 作為 loss function
    2. model/*.py 中輸出不必再通過 log_softmax()
    * 註解 : cross_entropy(x) ~= nll_loss( log_softmax(x) )
3. <新增> | train_args.py 中 Xray 部分
4. <更動> | Xray with VGG16 訓練參數
    * 使用epoch = 10, lr = 0.01, momentum = 0.9, image_size = ( 256, 256 ) 訓練通過

#### "Edward1997 09/06 17:20"
1. <更動> | "VGGNet.py" : 
實現 VGG11、VGG13、VGG16、VGG19 的模型
改用全連接層作為分類器，使其對應到目標的分類個數上 ( 原為使用多層卷積層，訓練不起來問題可能是出在此處 )
目前測試用 VGG16 訓練 MNIST 成功，但測試 Xray 常出現顯卡記憶體不足的狀況，VGG 模型還是太占記憶體空間了
2. <新增> | "data/Xray/Xray_preprocessing.py" : 
    1. resize() : convert (1024, 1024) to (256, 256) 
    2. to_gray() : convert (1024, 1024, 4) to (1024, 1204)
    3. delete_multi_label() : delete multi-label images
    4. balance() : down sampling the selected directory
    5. overview() : check the labels of images in the selected directory

#### "uuuChen 09/04 23:30"
1. <更動> | 將原本在 "local_central_train/"、"local_split_train/"、
"server_agent_train/"的檔案刪除，以 "local_central_train.py"、
"local_split_train.py"、"Agent_1.py"、"Agent_2.py"、"Agent_3.py"、"Agent_4.py"
、"Server.py" 替代，提升整體的簡潔性

#### "uuuChen 09/04 22:00"
1. <除錯> | "server.py", "agent.py": 原先直接將 model, optimizer 傳給下個
agent，更改為傳 model.stage_dict(), optim.state_dict()。更動後正確率與
central train 相近許多
 
#### "uuuChen 09/02 15:00"
1. <更動> | "server.py", "agent.py": 訓練時完全比照 "local_split_train.py" 
程式碼，然而正確率效果並不如 "local_split_train.py" 
 
#### "uuuChen 09/02 13:00"
1. <更動> | "data_processor.py": "將 db_id_list" 更名為 "usage_data_ids" 

#### "uuuChen 09/02 02:30"
1. <更動> | "server.py": 整理程式碼，更加易讀，功能不變
2. <更動> | "agent.py": 整理程式碼，更加易讀，連接下一個 agent 時除了 model，
也會把 optimizer 傳過去

#### "uuuChen 09/02 01:00"
1. <新增> | "central.py": 根據 data_name 選取對應的 dataSet, train_args, 
model，並實作 "get_model()", "get_dataSet()", "get_train_args()"等 相關
function

2. <新增> | "local_central_train.py": 匯集各 dataSet 的 "local_central_train"，
傳入 data_name ，再呼叫 "start_training()" 便開始訓練

3. <新增> | "local_split_train.py": 匯集各 dataSet 的 "local_split_train"，
傳入 data_name ，再呼叫 "start_training()" 便開始訓練

4. <新增> | "train_args.py": 匯集各 dataSet 的 train_args

#### "uuuChen 08/31 18:00"

簡化 "server.py", "agent.py" 程式

1. <更動> | "server.py": 將 "train", "test" 程式碼合併
2. <更動> | "agent.py": 將 "train", "test" 程式碼合併，並且 
"get_prev_next_agent_attrs" 改成只運行一次，之後就不再從 server 獲得 

#### "uuuChen 08/31 12:30"
<br>
在 "MNIST_Server_train.py" 裡增加 "is_simulate" 變數，此值為 True 的話
表示進行正確率模擬，假設每一個 agent 都擁有所有的 data, labels 進行訓練；
此值為 False 表示真實使用醫院資料，假設每一個醫院擁有各自的 data, labels
進行訓練。

1. <新增> | "server.py" 中 "get_total_data_nums_from_first_agent()": 在
每一個 agent 都有所有的 data 的前提下，使用此 function 拿到所有資料數

2. <新增> | "server.py" 中 "send_id_lists_to_agents()": 由 server 分配
每一個 agent 分到的 id_list。假設總共有 7 筆資料、 4 個玩家，分配的方式為
前三個玩家分到 2 筆資料，第四個玩家分到1筆資料

3. <更動> | "agent.py" 中 "start_training()": 配合 server 的更動進行調整

4. <新增> | "MNIST_Server_train.py", "agent.py": 新增 is_simulate 
bool 變數，此值為 True 的話表示進行正確率模擬，假設每一個 agent 都擁有所有
的 data, labels 進行訓練；值為 False 表示真實使用醫院資料，假設每一個醫院
擁有各自的 data, labels 進行訓練

5. <更動> | "data_processor.py" 中 "_get_data_and_labels_from_database()"
: 更動 data_nums 的獲得方式，將不再直接取得整個 database 的資料數，考慮
到模擬時真實的訓練數量是由 server 分配，因此以 "usage_data_ids" 為判斷訓練資料
數量的依據


#### "Edward1997 08/28 11:30"
1. <新增> | "server.py"
將所有 server 會使用到的共用功能打包
可藉由 train_args 傳入要訓練的 model 與 dataSet
2. <更動> | "agent.py"
藉由 server 傳來之 train_args 中的 dataSet 參數去取用對應資料集
並更動相關參數讀取與傳送順序
3. <新增> | "agent.py" 中 get_dataSet()、send_data_nums()、training_setting()
為了整理版面，將部分程式打包成函數
get_dataSet() : 根據train_args 中的 dataSet 參數去取用對應資料集
send_data_nums() : 將該資料集中資料個數寄給 server(train、test分開)
training_setting() : 設定 cuda 、 optimizer

#### "Edward1997 08/27 18:30"
1. <更動> | "MNIST_Server_train.py"
train_with_cur_agent 中 optimizer_server.zero_grad() 放於正確位置
解決上一版正確率下降問題
2. <更動> | "agent.py"
調整終止條件判定的寫法，解決最後一個agent不會自動關閉的情況
3. <更動、新增> | "MNIST_Server_train.py" 中 send_train_args()
增加傳送內容，傳送 agent 本身的IP(原為手動確認輸入)、port(原為手動分配輸入)
並調整 agent.py、MNIST_Agent_(1~4)_train.py 相關參數
4. <新增> | "MNIST_Server_train.py" 中 send_prev_next_agent_attrs()
server 寄送當前 agent 的前後 agents 給它，讓她知道該跟誰進行 snapshop

#### "Edward1997 08/27 15:40"
1. <更動> | "MNIST_Server_train.py", "server_agent_train/agent.py"
server 一次建立多組與 agent 的連線，
並在建立連線後收到 agents 的 data 個數，接著傳 train_args 給所有 agents

agent.py 中 _train_epoch 與 _test_epoch 合併為 _iter
MNIST_Server_train.py 可調整訓練 agent 個數、server port 起數數值、agent host port 起始數值(用於 snapshot)

然正確率有所下滑目前最好為89%(一對二,epoch 5)，不如集中式的97%，尋找問題中
帶補充的功能有；server 儲存 agents 的 label
2. <新增> | "server_agent_train/agent.py" 中 send_model()、get_prev_model()
snapshop 所用之功能
3. <新增> |  "MNIST_Server_train.py" 中 send_train_args()、get_data_nums()
server 傳送 train_args，並接收 agent 中資料各自了數量(train、test 分開)
4. <新增> |  "MNIST_Server_train.py" 中 conn_to_agents()、train_with_cur_agent()、test_with_cur_agent()
conn_to_agents() : server 與設定好個數的 agents 連線
train_with_cur_agent() : 與某一個 agent 進行訓練
test_with_cur_agent() : 與某一個 agent 進行測試
5.<更動> | "central_dataset_import.py"
更正名稱拼字錯誤

#### "uuuChen 08/26 12:00"
1. <新增> | "server_agent_train/agent.py": 因為各個 agent 的程式碼重複率很
高，所以把它獨立寫成一個 class ，大幅減少程式碼
2. <更動> | "socket_/socket_.py" 裡 "send()", "recv()": 在 "send()"
的時候會先傳送 header 再傳送 data ，中間間隔 time.sleep(0.1)，更動為 sleep()，
而在 "reve()" 到 header 後 "awake"
3. <更動> | "MNIST_Agent_1_train.py": 使用 class "Agent" 改寫
4. <更動> | "MNIST_Agent_2_train.py": 使用 class "Agent" 改寫
5. <更動> | "MNIST_Agent_3_train.py": 使用 class "Agent" 改寫
6. <更動> | "MNIST_Agent_4_train.py": 使用 class "Agent" 改寫

#### "Edward1997 08/22 13:40"
1. <新增> | "VGGNet.py":
嘗試使用 VGGNet 於 MNIST 但效果不彰，先更新上來以作備用

#### "uuuChen 08/20 16:30"
1. <更動> | "MNIST_Server_train.py": 詳見 HackMD "8/22 Weekly Meeting"
1. <更動> | "MNIST_Agent_1_train.py": 詳見 HackMD "8/22 Weekly Meeting"
1. <新增> | "MNIST_Agent_2_train.py": 詳見 HackMD "8/22 Weekly Meeting"
1. <新增> | "MNIST_Agent_3_train.py": 詳見 HackMD "8/22 Weekly Meeting"
2. <新增> | "MNIST_Agent_4_train.py": 詳見 HackMD "8/22 Weekly Meeting"
3. <新增> | "socket_.py" 中 "is_right_conn": 詳見 HackMD "8/22 Weekly Meeting"
4. <新增> | "socket_.py" 中 "awake": 詳見 HackMD "8/22 Weekly Meeting"
5. <新增> | "socket_.py" 中 "sleep": 詳見 HackMD "8/22 Weekly Meeting"

#### "Edward1997 08/19 11:20"
1. <除錯> | "data_processor.py" 裡 "_get_data_and_labels_from_database" :
若剩餘資料量 < batch，則只取到資料尾端，不再使用資料集開頭資料補上，避免開頭資料二次利用使訓練或測試不公平
2. <更動> | "MNIST_Server_train.py", "MNIST_Agent_train.py" :
將主導權交還 Server_train，train_arge 發配以及 loss 呈現等功能改為 Server_train 處理

#### "uuuChen 08/16 12:25"
1. <新增、除錯> | 新增 "socket_/socket_.py" 裡 "send()", "_send()", 
"recv()",  "_recv()" 等函式。在 window 上支援一次接收 "100000 bytes" 的
大封包，但是在 MACBOOK 上測試會出現問題。而新增這些函式後便也能在 MACBOOK 支援 
大封包傳送與接收，且測試結果正確。之後會在 LINUX 系統上進行測試。
 
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
"data_processor.py" 新增 "drop_coll_from_database()"，根據 use_gridFS
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
