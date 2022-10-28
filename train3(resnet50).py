import os
import numpy as np
import tensorflow as tf
import time
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#텐서플로우 가이드 지침 알림 출력 끄기...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# python = 3.7.11
#필요 라이브러리 목록
# tensorflow: 2.8.0 (pip install tensorflow==2.8.0)
# numpy: 1.21.5 (conda install numpy==1.21.5)
# sklearn: 1.0.2 (pip install sklearn==1.0.2)

#각종 변수를 따로 지정하여 실행할 수 있으나 기본적으로 지정되어 있기에
#따로 값을 지정하지 않아도 실행 할 수 있게 하였습니다

# 실행문 - python "train3(resnet50).py"
# -bs = 배치사이즈
# -er = 목표 에러율 해당 에러율이 될때까지 반복하게 되지만 현재 코드에서는 쓰지 않는상태입니다.
# -ep = 해당 값만큼 반복하고 로그를 출력
# -cl = Conv2d Layer 개수 한 계층당 2개의 conv 레이어 포함하며 기본으로 한개가 존재 (ex clayer = 2 -> 1 + 2 * 2 - 6개) 또한 pooling 계층 포함
# -img = 학습하게 될 이미지 사이즈
# -u = 학습 간격간에 유닛 크기 배율
# -p = 해당 모델 이름을 받게되며 해당 모델이름을 통해 폴더를 생성 후 해당 폴더에 모델과 
# -a = 기본값은 train으로 학습을 1회 진행하게 되며 
#              test를 입력하게 되면 학습을 하지 않고 데이터셋에 일부분을 받아와 테스트를 진행후 models/'path'에 테스트 기록을 저장하게 됩니다.

#시간을 출력하기 위한 클래스
class PrTime:
    initTime = None
    
    #프로그램 시작 시간 설정
    def __init__(self) -> None:
        self.initTime = time.time()
    
    #현재 시간에서 시작 시간을 뺀다.
    def retTimeBegin(self):
        beginTime = "({:0.1f}s)".format(time.time() - self.initTime)
        return beginTime

#경로를 각각 클래스 안에 저장하기 위해 따로 분리
class DirPath:
    path = '' #기본 경로
    modelPath = '' # 기본경로 + 모델 저장폴더
    batchSize = 0 # 배치사이즈 폴더 이름에 붙이기 위한 변수
    img_size = None
    
    #레이어 개수, 배치사이즈 크기를 경로에 추가 및 배치사이즈 변수 설정
    def __init__(self, layer, batchSize, unit, img_size) -> None:
        path = '/models/'
        self.path = path
        self.img_size = img_size
        self.batchSize = batchSize

    #폴더가 있으면 만들지 않으며 경로가 존재하지 않으면 경로(여러개의 폴더) 생성
    def MakeDirectory(self, fName):
        #모델 폴더까지 한번에 생성하기 위한 변수
        modelPath = './{}/{}'.format(self.path, fName)
        self.modelPath = modelPath # 재활용을 위한 저장
        try:
            #경로가 존재하는가?
            if not os.path.exists(self.modelPath):
                os.makedirs(self.modelPath) # 존재하지 않으면 한번에 생성
        except OSError:
            print('Error: Creating Directory Failed' + self.modelPath)

    #입력받은 이름의 파일을 열거나 생성하며 dataList의 내용을 한줄씩 입력한다
    def OutLog(self, fName, dataList):
        fN = "{0}/{1}_Log_BatchSize{2}_imgsize{3}.txt".format(self.modelPath, fName, self.batchSize, self.img_size)
        if os.path.isfile(fN) and fName=='Train':
            file = open(fN, 'a')
        else:
            file = open(fN, 'w')
        
        for line in dataList:
            file.write(line + '\n')

        file.close()
        
    #keras 모듈에 존재하는 모델 저장 함수        
    def SaveModel(self, model):
        model.save(self.modelPath)
        
    def LoadModel(self):
        model = tf.keras.models.load_model(self.modelPath)
        
        return model
    
    def ErrSave(self, err):
        fN = "{0}/{1}_Log_BatchSize{2}_imgsize{3}.txt".format(self.modelPath, 'Err', self.batchSize, self.img_size)
        file = open(fN, 'w')
        
        file.write(str(err))

        file.close()
        
    def ErrLoad(self):
        fN = "{0}/{1}_Log_BatchSize{2}_imgsize{3}.txt".format(self.modelPath, 'Err', self.batchSize, self.img_size)
        if os.path.isfile(fN):
            file = open(fN, 'r')
            err = (file.read()).split()[0]
        else:
            file = open(fN, 'w')
            err = 1.0
            file.write(str(err))
        file.close()
        return float(err)


class DNN: #slp 에서 중간에 여러개의 히든 레이어를 집어 넣어 DNN을 구현
    line = '------------------------------------------------------------------------------------------' #출력 구분을 위한 선
    model = None # 모든 레이어가 포함된 모델
    md = None
    t = None # 시간출력 클래스
    dp = None # 로그,모델 저장을 위한 클래스
    logList = [line] #로그 출력 및 저장을 위한 리스트 변수
    i = 0 # 몇번쨰 반복인가 카운트하는 변수
    epoch = 0 #몇번 반복마다 로그를 출력할 것인가를 지정하는 변수
    cLayer = 0 #합성곱 레이어 개수
    img_size = 32
    train_ds = None
    valid_ds = None
    unit = 32

    def __init__(self, train_ds, valid_ds, epoch, batchSize, errRate, cLayer, imgSize, unit, path) -> None:
        self.t = PrTime() # 시작 시간 지정
        self.cLayer = cLayer
        self.epoch = epoch # 로그 출력을 위한 반복지정
        self.batchSize = batchSize # 학습 변수인 배치사이즈 크기 지정
        self.img_size = imgSize
        self.dp = DirPath('50', batchSize, unit, imgSize) #경로 지정
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.unit = unit
        
        self.dp.MakeDirectory(path)#경로 생성
        
        self.ConvLayerAdd()#합성곱 계층 생성 (개수가 0 이상일 때)
        
        #로그 출력 포멧
        logLine = '{0}Layer | unit{1} | BatchSize{2} | size{3}'.format(path, unit, batchSize, imgSize)
        print(logLine) # 로그 출력
        print(self.line) # 로그 선
        self.logList.append(logLine) #로그 저장
        self.logList.append(self.line)
    
    def ConvLayerAdd(self):#합성 곱 계층을 cLayer 변수 크기의 개수만큼 생성하는 함수 활성 함수는 relu 사용
        if self.cLayer > 0:
            
            modelInput = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 1)) #입력 계층 생성
            self.md = tf.keras.layers.Rescaling(1./255)(modelInput)
            
            self.md = tf.keras.layers.Conv2D(self.unit, 
                                           (7, 7), 
                                           strides = (2, 2), 
                                           padding = 'same', 
                                           use_bias = False)(self.md)
            self.md = tf.keras.layers.BatchNormalization()(self.md)
            self.md = tf.keras.layers.Activation('relu')(self.md)
            
            self.md = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), 
                                                  strides=(2, 2), 
                                                  padding = 'same')(self.md)
            
            self.md = self.ConvBlock(self.md, 0)
            self.md = self.ConvBNIden(self.md, 0)
            self.md = self.ConvBNIden(self.md, 0)
            
            self.md = self.ConvBlock(self.md, 1)
            self.md = self.ConvBNIden(self.md, 1)
            self.md = self.ConvBNIden(self.md, 1)
            self.md = self.ConvBNIden(self.md, 1)
            
            self.md = self.ConvBlock(self.md, 2, 2)
            self.md = self.ConvBNIden(self.md, 2)
            self.md = self.ConvBNIden(self.md, 2)
            self.md = self.ConvBNIden(self.md, 2)
            self.md = self.ConvBNIden(self.md, 2)
            self.md = self.ConvBNIden(self.md, 2)
            
            self.md = self.ConvBlock(self.md, 3, 2)
            self.md = self.ConvBNIden(self.md, 3)
            self.md = self.ConvBNIden(self.md, 3)
            
            self.md = tf.keras.layers.GlobalAveragePooling2D()(self.md)
            self.md = tf.keras.layers.Dense(4096, activation = 'relu')(self.md)
            self.md = tf.keras.layers.Dropout(.5)(self.md)
            self.md = tf.keras.layers.Dense(4096, activation = 'relu')(self.md)
            self.md = tf.keras.layers.Dropout(.5)(self.md)
            modelOutput = tf.keras.layers.Dense(6, activation = 'softmax')(self.md)# 분류를 위한 soft max 함수를 통해 모두 합쳐 1에 값이 나오게 계산
            self.model = tf.keras.models.Model(modelInput, modelOutput)# 모델 생성

    def ConvBlock(self, x, units, stride=1):
        shortcut = x
        shortcut = tf.keras.layers.Conv2D((pow(2, units)*self.unit)*4, (1, 1), strides = (stride, stride), padding = 'valid', use_bias = False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        #
        x = tf.keras.layers.Conv2D(pow(2, units)*self.unit, (1, 1), strides = (stride, stride), padding = 'valid', use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(pow(2, units)*self.unit, (3, 3), padding = 'same', use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D((pow(2, units)*self.unit)*4, (1, 1),  padding = 'valid', use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
    
    def ConvBNIden(self, x, units):
        shortcut = x
        #
        x = tf.keras.layers.Conv2D(pow(2, units)*self.unit, (1, 1), padding = 'valid', use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(.1)(x)
        x = tf.keras.layers.Conv2D(pow(2, units)*self.unit, (3, 3), padding = 'same', use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(.1)(x)
        x = tf.keras.layers.Conv2D((pow(2, units)*self.unit)*4, (1, 1),  padding = 'valid', use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # 
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
    
    
    def CompModel(self, optimizer, loss, metrics): # 모델 학습을 위한 구성 지정
        self.model.compile(optimizer = optimizer, #
                           loss = loss, 
                           metrics = metrics)
    
    #모델 학습 명령어
    def fitModel(self):
        self.i += self.epoch #몇번 반복했는가
        self.model.fit(self.train_ds, 
                       validation_data = self.valid_ds, 
                       epochs = self.epoch #한번 실행해서 몇번 반복할 것인가
                       )
    
    #추가로 학습할려 할 때 사용하는 함수
    def TrainOnBatch(self, x, y):
        self.i += 1
        self.model.train_on_batch(x, y)
    
    #모델 계층 구조 png로 출력
    def PlotModel(self, path):
        tf.keras.utils.plot_model(self.model, './models/{}/model.png'.format(path))

    #모델 에러율 검출 및 로그를 리스트에 저장
    def EvalModel(self, name):
        loss, accuracy = self.model.evaluate(self.valid_ds, verbose = 0)
        errorRate = 1.0 - accuracy
        testLog = '{0} | Iteration-{1}  |  ErrorRate-{2:0.2f}%  |  Time-{3}'.format(name, self.i, errorRate * 100, self.t.retTimeBegin())
        self.logList.append(testLog)
        
        return errorRate
    
    #모델, 로그를 해당되는 폴더에 저장
    def OutTrainData(self):
        self.dp.OutLog('Train', self.logList)#경로에 Train 로그 저장
    
    def saveModel(self):
        self.dp.SaveModel(self.model)#경로에 모델 저장
    
    #총 결과 로그를 출력하고 리스트 변수에 저장
    def TotalLog(self, err):
        logLine = 'Time-{0}  |  Iteration-{1}  |  Total Train ErrorRate = {2:0.2f}'.format(self.t.retTimeBegin(), self.i, err*100)
        print(logLine)
        self.logList.insert(0, logLine)
    
    #테스트용 함수 학습이 끝난 모델을 테스트 데이터를 통해 평가하고 로그를 텍스트로 남긴다.
    def OutTestLog(self, x):
        testList = []
        t = self.t.retTimeBegin()
        wrong = 0.0
        label = ['20대(여)', '30대(여)','40대(여)', '20대(남)', '30대(남)', '40대(남)']
        
        y1 = np.empty((0,6))
        y2 = np.empty((0,6))

        for data in x:
            img = data[0]
            labels = data[1]
            p = self.model.predict(img)
            y1 = np.append(y1, p, axis=0)
            y2 = np.append(y2, labels, axis=0)
        
        y1 = np.argmax(y1, axis=1)
        y2 = np.argmax(y2, axis=1)
        total = y1.shape[0]
        for i, p in enumerate(y1):
            
            if y2[i] != p:
                wrong += 1.0
            
            testLine = 'TEST[{0:05d}]  |  Predict-{1}  |  Answer-{2}  |  ErrRate-{3:0.2f}%)'.format(i+1, label[p], label[y2[i]], (wrong/total) * 100)
            testList.append(testLine)
        
        testLine = 'Time-{0}  | Total Test ErrorRate-{1:0.2f}'.format(t, (wrong/total) * 100)
        print(self.line + "\n" + testLine + "\n" + self.line )
        testList.insert(0, testLine)
        
        testList.append('accuracy_score: {}'.format(accuracy_score(y2, y1)))
        testList.append('recall_score: {}'.format(recall_score(y2, y1, average='micro')))
        testList.append('precision_score: {}'.format(precision_score(y2, y1, average='micro')))
        testList.append('f1_score: {}'.format(f1_score(y2, y1, average='micro')))
        
        self.dp.OutLog('Test', testList)
        
    def getStrSummary(self):
        sumList = []
        self.model.summary(print_fn = lambda x: sumList.append(x))
        
        self.dp.OutLog('mdoel_summary', sumList)
        
        return '\n'.join(sumList)
    
    def getModel(self):
        self.model = self.dp.LoadModel()
        
    def saveErr(self, err):
        self.dp.ErrSave(err)
        
    def loadErr(self):
        return self.dp.ErrLoad()
    
    def saveparam(self, path):
        self.model.save_weights('./models/'+path+'/weights.h5', overwrite = True)

    def loadparam(self, path):
        self.model.load_weights('./models/'+path+'/weights.h5')


def main(batch_Size, finalErr, epoch, cLayer, image_size, unit, path, act):
    
    train_ds = None
    valid_ds = None
    test_ds = None
    
    if act == 'test':
        test_ds = tf.keras.utils.image_dataset_from_directory(
            './img', 
            label_mode = 'categorical', 
            color_mode = 'grayscale', 
            batch_size = batch_Size, 
            image_size = (32, 32), 
            seed = 3737, 
            validation_split = 0.01, 
            subset = 'validation'
        )
    elif act == 'train':
        train_ds = tf.keras.utils.image_dataset_from_directory(
            './img', 
            label_mode = 'categorical', 
            color_mode = 'grayscale', 
            batch_size = batch_Size, 
            image_size = (image_size, image_size), 
            shuffle = True, 
            seed = 3737, 
            validation_split = 0.2, 
            subset = 'training'
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            './img', 
            label_mode = 'categorical', 
            color_mode = 'grayscale', 
            batch_size = batch_Size, 
            image_size = (image_size, image_size), 
            shuffle = True, 
            seed = 3737, 
            validation_split = 0.2, 
            subset = 'validation'
        )
    else:
        print('act를 정확히 입력해주세요 - test or train')
        exit()
    
    dnn = DNN(train_ds, valid_ds, epoch, batch_Size, finalErr, cLayer, image_size, unit, path)
    
    del(train_ds)
    del(valid_ds)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001)
    
    dnn.CompModel(optimizer, 'categorical_crossentropy', ['accuracy'])
    err = 1.0
    
    if len(os.listdir('./models/{}'.format(path))) > 0:
        dnn.loadparam(path)
        err = dnn.loadErr()
    
    summary = dnn.getStrSummary() #모델 계층마다의 파라미터 개수 저장 및 반환
    
    if act == 'test':
        dnn.OutTestLog(test_ds)
    else: 
        summary = dnn.getStrSummary() #모델 계층마다의 파라미터 개수 저장 및 반환
        for i in range(1):
            dnn.fitModel()
            errRate= dnn.EvalModel('Validation')
            if err > errRate:
                dnn.saveparam(path)
                dnn.saveErr(errRate)
                err = errRate
    
    dnn.OutTrainData()
    dnn.PlotModel(path) #모델 구조 이미지 저장

if __name__ == "__main__":
    #배치스크립트를 실행하기 위한 arguments를 받아오는 부분 없으면 기본값으로 실행이 됩니다.
    parser = argparse.ArgumentParser(description='RESNET50 구현')
    parser.add_argument('-bs', '--batch_size', help = '배치 사이즈', type=int, default=128)
    parser.add_argument('-er', '--final_err', help = '최종 도달 목표 에러율', type=float, default=0.10)
    parser.add_argument('-ep', '--epoch', help = '해당 값만큼 반복하고 로그를 출력', type=int, default=1)
    parser.add_argument('-cl', '--c_layer', help = '은닉 계층 개수(conv2d,conv2d,pooling)', type=int, default=2)
    parser.add_argument('-img', '--image_size', help = '이미지 사이즈', type=int, default=32)
    parser.add_argument('-u', '--unit', help = '한 계층당 유닛 크기 배율', type=int, default=64)
    parser.add_argument('-p', '--path', help = '모델 저장 경로', type=str, default='resnet50')
    parser.add_argument('-a', '--act', help = '학습할지 테스트 할지', type=str, default='train')
    args = parser.parse_args()
    
    main(args.batch_size, args.final_err, args.epoch, args.c_layer, args.image_size, args.unit, args.path, args.act)