## Research Specification

* [分散可擴充式醫療資料跨域 AI 運算整合方案 – 業務委託內容](https://drive.google.com/open?id=1r2ae91fA65ef5zaA3wWoUk2LXBlrinEI)


## Reference
* [Distributed learning of deep neural network over multiple agents](https://www.sciencedirect.com/science/article/pii/S1084804518301590)

## Distributed Learning Architecture
* ![](https://i.imgur.com/7JS5mHh.jpg)
* ![](https://i.imgur.com/hXDnbQo.png)

### Distributed Architecture with Docker
* ![](https://i.imgur.com/ytAX8ev.png)

## Technical Report Outline 

* **分散式架構系統建置流程技術報告 (含原始碼)** - [**Notebook**](/RLUYo07VQZKQm1cunqQZSw)
    * 1. 環境建置-開發框架與套件安裝 (集中式、分散式) (例: Anaconda/Pytorch)
    * 2. 環境建置-資料庫環境安裝 (集中式、分散式) (例: MongoDB & PyMongo) 
    * 3. 分散式架構 - 建構資料端 (Agents) 與模型端 (Server) 之間的通訊 API (8/22)
    * 附件: 環境建置流程相關原始碼 
* **分散式系統架構驗證與效能測試技術報告 (含原始碼)** - [**Notebook**](/rhaaN2klTQiRNA5dqtSKuw)
    * 1. 測試模型架構與對應資料集準備 (4種)(8/8)
        * 1-1. MNIST / LeNet
        * 1-2. [Diabetic Retinopathy Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) / AlexNet
        * 1-3. [NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data) / VGG16
        * 1-4. [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat) / MLP
    * 2. 使用 PyMongo 進行訓練資料與開發框架之間的串接(8/8)
    * 3. 傳統集中式訓練環境效能測試(9/5)
    * 4. 分散式架構訓練環境效能測試 ( 2, 3, 4 agents )
        * 4-1. MNIST + LeNet ( 30 epochs )
        * 4-2. MC + AlexNet ( 50 epochs )
        * 4-3. OCT + VGG ( 50 epochs )
        * 4-4. ECG + MLP ( 30 epochs )
    * 5. 可切割模型設計差異之效能比較 (以 4 agents 為例)(9/19)
        * 5-1. Agent 端模型層數 > Server 端模型層數
        * 5-2. Agent 端模型層數 < Server 端模型層數
    * 附件: 效能測試相關原始碼
