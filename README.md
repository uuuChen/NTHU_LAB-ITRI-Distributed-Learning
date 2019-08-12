# distributed_learning
--------------------------------------------
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

#### "cleat README.md updatalist till 08/08"  | Edward1997 | 08/12

#### "vim mongoDB_processor.py"  | Edward1997 | 08/12
* def gridFS_coll_insert
	label = int(labels[i]) 
	-->	label = labels[i]
* def gridFS_coll_read_batch
	batch_labels.append(label)
	--->	label  = np.array(label)
        	batch_labels.append(label)

#### "vim DRD_train"  | Edward1997 | 08/12
* def train_epoch
	data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size)
	-->	data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size, image_size= train_args.image_size)

#### "vim DRD_train"  | Edward1997 | 08/12
* def test_epoch
	data, target = test_dataSet.get_data_and_labels(batch_size=train_args.batch_size)
	-->	data, target = train_dataSet.get_data_and_labels(batch_size=train_args.batch_size, image_size= train_args.image_size)

#### "add delete_all_database.py"  | Edward1997 | 08/12

#### "add centrl_dataset_import.py"  | Edward1997 | 08/12

#### "add Xray_dataset.py"  | Edward1997 | 08/12

#### "vim MNIST_train.py, DRD_train.py"  | Edward1997 | 08/12
* def train_epoch
	datas_tratned_num += len(data)
	避免已訓練資料顯示超過總資料上限
--------------------------------------------
