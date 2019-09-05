# しったかChainerCV～ResNetで画像識別してみよう～


@<icon>{yousei} 「前回はChainerCVを使って画像処理の基礎についてお話したよね。覚えている？」

　

@<icon>{cheiko} 「もちろんよ。画像のチャネル構成や、画像変形、水増し、データセットについて学んだし復習もしたから大丈夫よ。」

　

@<icon>{yousei} 「じゃあ今回は、ちょっと進めてDeepLearningモデルであるResNetを使って画像分類をしてみよう。ChainerCVを使うと簡単に実装できることがわかるよ。」

　

@<icon>{cheiko} 「れずねっと？」

　

@<icon>{cheita} 「ResNetだよ。DeepLearning界隈で有名なImageNetの分類問題のコンペであるILSVRC 2015でエラー率3.57%となり、1位を獲得したモデルさ。正式にはResidualNetといい、残差を・・・」

　

@<icon>{cheiko} 「・・・（またはじまったわ。この子）」

　

@<icon>{yousei} 「よく知っているね。今日はその高精度なモデルResNetであっても実装は簡単だということを見せてあげよう。」

　
　

先生の言う通り、DeepLearningで画像分類と言ってもChainerCVを使えば難しくはありません。手順どおりに進めていくだけでできちゃいます。なにはともあれChainerCVをインストールしてはじめよう！


```
!pip install chainercv==0.12
```

#### 注意

* この章ではGoogleColabratory@<fn>{fn01}上で実行することを想定しています。
* GoogleColabratoryではGPUを使うように設定してください。メニューの[ランタイム]-[ランタイムのタイプを変更]で[ハードウェアアクセラレータ]項目に[GPU]を設定してください。
* 執筆時点でGoogleColabratoryはChainer 5.4.0がインストールされているため合わせてChainerCVは0.12を指定しています。ChainerとChainerCVとの組み合わせはhttps://github.com/chainer/chainercv を参照してください。

//footnote[fn01][https://colab.research.google.com/]



## 画像データの準備


最初に画像分類するための画像を用意しましょう。めんどくさい画像収集もChainerCVを使えば、画像データセットを数行で取得できる仕組みがあります。

```
import numpy as np
import random

import chainer
from chainercv.datasets import OnlineProductsDataset
from chainercv.datasets import online_products_super_label_names as label_names

dataset_train = OnlineProductsDataset(split='train')
dataset_test = OnlineProductsDataset(split='test')
```

上記のコードだけで学習用画像データセットとテスト用画像データセットが取得できます。学習用データセットを使いモデルを構築し、テスト用データセットを使い構築したモデルを評価する。というように使います。そのため学習用画像データとテスト用画像データで画像が混ざってはいけません。それではカンニングになってしまいますからね。

　

さて、どれくらいのデータ量なのか確認してみましょう。

```
print("size:{}".format(len(dataset_train))) # -> size:59551
print("size:{}".format(len(dataset_test)))  # -> size:60502
```

上記のコードを実行すると、それぞれ約6万件ぐらいはあることがわかりますね。次にそのデータセットがどのような画像で、それを意味するラベルに何がついているのか確認してみます。

```
from chainercv.transforms import resize
import matplotlib.pyplot as plt

def view_dataset_samples(split_name, dataset):
  fig = plt.figure(figsize=(12, 3))
  fig.suptitle('sample dataset for {}'.format(split_name), fontsize=20)
  fig.tight_layout()
  for i in range(5):
      _data = random.choice(dataset_train)
      _image = resize(_data[0], (200, 200))
      _image = _image.transpose((1, 2, 0))  # CHW -> HWC
      _title = "{}:{}".format(_data[2], label_names[_data[2]])

      ax = fig.add_subplot(1, 5, i+1)
      ax.imshow(_image.astype(np.uint8))
      ax.set_axis_off()
      ax.set_title(_title)

view_dataset_samples('train', dataset_train)
view_dataset_samples('test', dataset_test)
```


![データセットのサンプル](src/images/chug_03_dataset_sample.png)

この画像データセットはStanford大学が公開しているOnline Products dataset@<fn>{fn02}です。オンライン販売のEbayに登録されている商品を12個に分類(bicycle, cabinet, chainer, coffe_maker, fan, kettle, lamp, mug, sofa, stapler, table, toaster)したデータです。


上の画像を見ると、画像とその分類名がセットになっています。このセットがいわゆる教師データとなります。画像と分類をセットで学習し適切なモデルをDeepLearningで構築します。その後、画像のみを与えて正しく分類が推論できるかどうかを評価することになります。

//footnote[fn02][http://cvgl.stanford.edu/projects/lifted_struct/]


## 画像変換クラスの作成

次に画像変換クラスを作成しましょう。ここでは主に画像の前処理について実装します。今回はscaleとcenter_cropを使います。

* scale

 * これは、画像を縦横を指定する同一長にリサイズする処理です。
 * 実は今回のデータセットの画像は、個々にサイズが異なるため一定のサイズにしたいのです。

* center_crop

 * これは、画像の中心を起点に指定サイズで切り取る(トリミング)処理です。
 * scaleで同一サイズにした画像をResNetモデルが処理できるサイズ(224,224)に変換したいのです。scale時に(224,224)を指定してもよいのですが、そもそも個々のサイズが異るため、対象物が中央に配置されないだろうことを危惧しての実装しています。
 

```
from chainercv.transforms import scale
from chainercv.transforms import center_crop

class ImageTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, _, label = in_data
        img = scale(img, 256)
        img = center_crop(img, (224, 224))
        img -= self.mean
        return img, label
```



## モデルを実装してみよう

いよいよDeepLearningのコアであるResNetモデルの実装です。とは言ってもChainerCVはResNetモデルがすでに実装されています。なので次のように数行のコードを書くだけでいいのです。なんと簡単っ！


```
from chainercv.links import ResNet50
from chainer.optimizers import Adam
from chainer.links import Classifier

extractor = ResNet50(n_class=len(label_names), **{'arch': 'fb'})
extractor.pick = 'fc6'
model = Classifier(extractor)

optimizer = Adam()
optimizer.setup(model)
```

ChainerCVを使えば、DeepLearning界隈で有名な各種モデルを自分で実装することなく簡単に利用することが出来るのです。さらにこれらはコントリビューターによって随時アップデートされ続けているのです。

　

次のURLにはChainerCVが実装されているモデルについて説明されています。

https://chainercv.readthedocs.io/en/v0.12.0/reference/links.html#model



## ランダムシードの固定化とGPU利用設定

DeepLearningでは内部の各所では、ランダム値が使われています。そのためモデルやパラメータを修正して評価するためにはランダムシードを固定させないと正しい評価はできません。そのための設定です。GPU利用設定も含めてそんなものだな。と理解していただければ結構です。

```
# ランダムシードの固定化
random.seed(0)
np.random.seed(0)
if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(0)
    
# GPUの利用設定
if chainer.cuda.available == True:
  GPUID=0
  xp = chainer.cuda.cupy
  chainer.cuda.Device(GPUID).use()
  model.to_gpu(GPUID)
else:
  GPUID=-1
  xp = np
```


## 学習用のデータセットを分割して前処理を組み込む

ここでは、学習用データセットを8:2で分割します。一つは本当に学習用として使うセット、もう一つはバリデーション用として使います。今回は問題を簡単にするためにバリデーションについては触れませんが、テスト用データセットとは異なり学習をより効率的に進めるための評価データセットがバリデーション用データセットです。

次に、さきほど作成した前処理の実装を組み入れます。これはTransoformDatasetクラスに包めるだけです。
　

```
from chainer.datasets import TransformDataset
from chainer.datasets import sub_dataset

split_at = int(len(dataset_train) * 0.8)
train, valid = sub_dataset.split_dataset(dataset_train, split_at)

print('{}'.format(len(train))) #-> 47640
print('{}'.format(len(valid))) #-> 11911

train_data = TransformDataset(train, ImageTransform(extractor.mean))
valid_data = TransformDataset(valid, ImageTransform(extractor.mean))
```



## トレーナーを作成して、いざ学習スタート

最後に、DeepLearningの学習を効率よく実装できるようにChainerにはTrainerという仕組みが導入されています。構成としては


```
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainer.training import extensions

BATCHSIZE=64
EPOCH=10

# 学習用のイテレータ、モデルアップデータ、トレーナーを作成
train_iter = chainer.iterators.SerialIterator(train_data, BATCHSIZE)
updater = chainer.training.StandardUpdater(train_iter, optimizer, device=GPUID)
trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'), out='result')

# 各種トレーナー設定
trainer.extend(extensions.observe_lr(), trigger=(1, 'iteration'))
trainer.extend(extensions.LogReport(trigger=(0.1, 'epoch')))
trainer.extend(extensions.PrintReport(
    ['iteration', 'epoch', 'elapsed_time', 'lr', 'main/loss', 'main/accuracy']), 
    trigger=log_interval)

# 定期的に状態をシリアライズ（保存）する機能
trainer.extend(
    extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(
    extensions.snapshot_object(model.predictor, 
    filename='model_epoch-{.updater.epoch}'))

# 学習スタート
trainer.run()
```

出力されるログ

```
iteration   epoch       elapsed_time  lr          main/loss   main/accuracy
75          0           131.208       0.000204978  2.32312     0.206042       
149         0           250.938       0.000324627  2.14418     0.227829       
224         0           372.256       0.000412265  2.05128     0.276042       

～　中略　～

2978        4           4892.57       0.000973282  1.15191     0.612753       
3052        4           5032.87       0.000975213  1.13557     0.616976       
3127        4           5154.62       0.000977015  1.14132     0.620833       

～　中略　～

7295        9           11955.2       0.000999649  0.694693    0.766047       
7370        9           12075.5       0.000999674  0.704617    0.764583       
7444        10          12194.4       0.000999698  0.714064    0.753167       
```



## 評価

テストする。


## さいごに

いかがでしたでしょうか。DeepLearningといっても難しく考えることはありません。まずはこれらのコードを写経して自分でも試してみてください。そうすることで、少しずつ自分のスキルとして蓄積されていきます。

今回は画像分類をDeepLearningで簡単に実装しました。本来はもうちょっと考慮する部分があります。例えば。。。

* バリデーション用データセット、テスト用データセットでの評価
* Optiomizerの検証
* 汎化性能の測定と対策
* ハイパーパラメータの調整

ですね。次回はもうちょっと奥深く進んでいきましょうね。



