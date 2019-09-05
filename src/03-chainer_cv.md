# しったかChainerCV～ResNetで画像分類～


@<icon>{yousei} 「前回はChainerCVを使って画像処理の基礎について学んだよね。覚えているかな？」

//blankline

@<icon>{cheiko} 「もちろんよ。画像のチャネル構成や、画像変形、水増し、データセットについて学んだし復習もしたから大丈夫よ。」

//blankline

@<icon>{yousei} 「じゃあ今回は、ちょっと進めてDeepLearningモデルであるResNetを使って画像分類をしてみよう。ChainerCVを使うと簡単に実装できることがわかるよ。」

//blankline

@<icon>{cheiko} 「れずねっと？」

//blankline

@<icon>{cheita} 「ResNetだよ。DeepLearning界隈で有名なImageNetの分類問題のコンペであるILSVRC 2015でエラー率3.57%となり、1位を獲得したモデルさ。正式にはResidualNetといい、残差を・・・」

//blankline

@<icon>{cheiko} 「・・・（またはじまったわ。この子）」

//blankline

@<icon>{yousei} 「よく知っているね。今日はその高精度なモデルResNetであっても実装は簡単だということを見せてあげよう。」

//blankline

先生の言う通り、DeepLearningで画像分類と言ってもChainerCVを使えば難しくはありません。手順どおりに進めていくだけでできちゃいます。なにはともあれChainerCVをインストールしてはじめてみよう！


```
!pip install chainercv==0.12
```

#### 注意

* この章ではGoogleColabratory@<fn>{fn01}上で実行することを想定しています。
* GoogleColabratoryではGPUを使うように設定してください。メニューの[ランタイム]-[ランタイムのタイプを変更]で[ハードウェアアクセラレータ]項目で[GPU]を設定できます。
* 執筆時点でGoogleColabratoryはChainer 5.4.0がインストールされているため合わせてChainerCVは0.12を指定しています。ChainerとChainerCVとの組み合わせはhttps://github.com/chainer/chainercv を参照してください。

//footnote[fn01][https://colab.research.google.com/]



## 画像データの準備


まずは、分類するための画像データセットを用意しましょう。めんどくさい画像収集もChainerCVを使えば、数行で取得できます。

```
import numpy as np
import random

import chainer
from chainercv.datasets import OnlineProductsDataset
from chainercv.datasets import online_products_super_label_names as label_names

dataset_train = OnlineProductsDataset(split='train')
dataset_test = OnlineProductsDataset(split='test')
```

上記のコードだけで学習用画像データセットとテスト用画像データセットが取得できます。それぞれ学習用データセットを使いモデルを構築し、テスト用データセットを使い構築したモデルを評価する。というように使います。そのため学習用画像データとテスト用画像データで画像が混ざってはいけません。それではカンニングになってしまいますからね。

//blankline

さて、どれくらいのデータ量なのか確認してみましょう。

```
print("size:{}".format(len(dataset_train))) # -> size:59551
print("size:{}".format(len(dataset_test)))  # -> size:60502
```

それぞれ約6万件ぐらいはあることがわかりますね。次にそのデータセットがどのような画像で、それを意味する分類名に何がついているのか確認してみます。

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


![データセットのサンプル](src/images/chainercv_dataset_sample.png)

この画像データセットはStanford大学が公開しているOnline Products dataset@<fn>{fn02}です。オンライン販売のEbayに登録されている商品を12個に分類(bicycle, cabinet, chainer, coffe_maker, fan, kettle, lamp, mug, sofa, stapler, table, toaster)したデータです。

//blankline

データセットを見ると、画像にはその分類名が付与されています。このセットがいわゆる教師データとなります。画像と分類名をセットで学習し適切なモデルをDeepLearningで調整します。モデルができたら今度は画像のみを与えて正しく分類名が推論できるかを評価します。

//footnote[fn02][http://cvgl.stanford.edu/projects/lifted_struct/]


## 画像変換クラスの作成

次に画像変換クラスを作成します。ここではDeepLearningの前処理を実装することになります。今回はscaleとcenter_cropを使いましょう。

* scale

 * これは、画像を縦横を同一長にリサイズする処理です。
 * 今回のデータセットは、個々に画像サイズが異なるのです。そのため全画像を一定のサイズにします。

* center_crop

 * これは、画像の中心を起点に指定サイズで切り取る(トリミング)処理です。
 * scaleで同一サイズにした画像をResNetモデルが処理できるサイズ(224,224)に変換したいのです。scale時に(224,224)を指定してもよいのですが、前述の通りそもそも個々のサイズが異るため、対象物が中央に配置されないだろうことを心配しての実装です。
 

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



## モデルの実装

いよいよDeepLearningのキモであるResNetモデルの実装です。とは言っても難しくありません。ChainerCVにはResNetがすでに実装されているのです。なので数行のコードを書くだけでいいのです。なんと簡単っ！


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

ChainerCVを使えば、DeepLearning界隈で有名な各種モデルを自分で実装することなくライブラリのように簡単に利用できます。さらにこれらはコントリビューターによって随時アップデートされ続けているのです。

　

次のURLにはChainerCVが実装されているモデルについて説明されています。

https://chainercv.readthedocs.io/en/v0.12.0/reference/links.html#model



## ランダムシードとGPU設定

DeepLearningの内部ではランダム値が多く使われています。そのためモデルやパラメータを変えた結果を評価するためにはランダムシードを固定させる必要があります。そのための設定です。ここではGPU利用設定も含めて、まぁそんなものだな。と考えていただければ結構です。

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


## 学習データの分割と前処理の組込み

次に、学習用データセットを8:2で分割します。一つは本当に学習用として使うセット、もう一つはバリデーション用として使うセットです。今回は問題を簡単にするためにバリデーションについては触れませんが、前述のテスト用データセットとは異なり学習時に、より汎化性能を得るための評価データセットとなります。

//blankline

そして、さきほど作成した前処理の実装(ImageTransform)を組み入れます。TransoformDatasetクラスに指定するだけです。
　

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

さぁ、これで大体の準備が整いました。



## いざ学習スタート

DeepLearningを効率よく実装できるようにChainerにはTrainerという仕組みが導入されています。構成としては次の図のようなイメージです。

![トレーナーの構成](src/images/chainercv_trainer_structure.png)

それではトレーナーの設定も含めて実装してみましょう。


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

# 定期的に状態を保存する設定
trainer.extend(
    extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(
    extensions.snapshot_object(
        model.predictor, 
        filename='model_epoch-{.updater.epoch}'))

# 学習スタート
trainer.run()
```

さっそく実行してみましょう。学習が終わるには３、４時間ぐらいかかるのではないかと思います。学習中には、状況を示すログが次のように出力されます。

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

ここで大事なのは、ログが出力されるたびにmain/loss値が0.0に近づき、main/accuracyが1.0に近づくことが観測できるかどうかです。さっそくグラフにしてみましょう。

```
import json
import pandas as pd
with open('result/log') as f:
  result = pd.DataFrame(json.load(f)).interpolate()

result[['main/accuracy','main/loss']].plot()
```

![学習結果のグラフ](src/images/chainercv_graph.png)

* main/lossとは

 * 損失関数によって算出された「正解とどれくらい離れているかを表す数値」です。0.0に近づくほどロスが少なくなるので正解に近いことを意味します。

* main/accuracyとは

 * 正解率を表す数値です。1.0 に近いほど正解に近いことを意味します。

//blankline

グラフからは、main/accuracyは0.85ぐらいで収束しそうなのでまぁまぁですが、main/lossはもっと低い値で収束させたいですよね。





## テストデータで評価する

学習データセットで学習した結果、まぁまぁなモデルができたことはわかってきました。それではテストデータセットではどのような結果になるのか確認してみましょう。

```
from chainer.training import extensions
from chainer import serializers

predictor = ResNet50(n_class=len(label_names), **{'arch': 'fb'})
serializers.load_npz("result/model_epoch-10", predictor, strict=True)
model_predictor = Classifier(predictor)
model_predictor.to_gpu(GPUID)

split_at = int(len(dataset_test) * 0.3)
test, _ = sub_dataset.split_dataset(dataset_test, split_at)

test_data = TransformDataset(test, ImageTransform(predictor.mean))
test_iter = chainer.iterators.SerialIterator(test_data, BATCHSIZE, repeat=False)
test_evaluator = extensions.Evaluator(test_iter, model_predictor, device=GPUID)
test_results = test_evaluator()
print('Test accuracy:', test_results['main/accuracy'])
```

テストデータセット全体の30%のデータを使って評価してみました。

```
Test accuracy: 0.6325345
```


学習データセットを使った時と比べて性能は低いです。これが過学習(over fitting)という状態です。学習データセットに対してだけ性能がよく、テストデータセットに対しては性能が低いということです。まったく同じ学習データの画像しか入力されないのであれば、over fittingで良いのですが、一般的にはそうではありません。

//blankline

例えばカメラで撮影される画像には、光の映り込みや、被写体の角度、陰影、色合い等、さまざまな要素が混入する可能性が高いでしょう。これらを無視し、正しいものだけを認識するためには汎化性能が求められることになります。


## 最後に

いかがでしたでしょうか。DeepLearningといっても難しく考えることはありません。まずはコードを写経して自分でもぜひ試してみてください。手を動かすことで、少しずつ自分のスキルになっていきますよ。

//blankline

本来、画像分類では、本来はもうちょっとだけ考慮する部分があります。例えば・・・

* バリデーション用データセット、テスト用データセットでの評価
* Optiomizerの検証
* 汎化性能の測定と対策
* ハイパーパラメータの調整

次回はそのあたりを奥深く進んでいきましょう！


//blankline
//blankline

@<icon>{yousei} 「どうだったかな？ 」

//blankline

@<icon>{cheiko} 「だ、だいじょうぶよ。次回のために復習もしてつもりよ。」

//blankline

@<icon>{cheita} 「早く、次回にならないかなぁ。今回は楽勝だっからからなぁー。僕はOptimizerに興味があるんだよね。AdamかSGDか、learning-rateはどうしようかといつも悩むんだよなぁ・・・お姉ちゃんは知らないだろうけど。」

//blankline

@<icon>{cheiko} 「・・・（この子、ちょっとうざいわ）」

