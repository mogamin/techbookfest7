# ChainerCV で画像識別してみよう！

@<icon>{yousei} 「前回は、ChainerCVを使って画像処理の基礎についてお話しましたね。覚えている？」

　

@<icon>{cheiko} 「も、もちろんよ。チャネルの構成や、画像の変形、水増し、データセットについて学んだわ。覚えることがたくさんあって大変だったけど、なんとか理解したわ。」

　

@<icon>{cheita} 「まぁ、基礎だったからね。ちょっと物足りなかったかなぁ。（チラっ」

　
　
@<icon>{cheiko} 「・・・え？　なにこの子。うざいわ。」

　

@<icon>{yousei} 「そうだよね。では今回は画像を使ってDeepLearningしてみよう！」

　

DeepLearningと言っても難しくはありません。順に進めていくと不思議とできちゃういます。さぁなにはともあれChainerCVをインストール！


```
!pip install chainercv==0.12
```

#### 注意

* この章ではGoogleColabratory@<fn>{fn01}上で実行することを想定しています。
* 現時点でGoogleColabratoryはchainer 5.4.0がインストールされているため合わせてChainerCVは0.12としました。chainerとchainercvとのバージョンの組み合わせはhttps://github.com/chainer/chainercv を参照してください。

//footnote[fn01][https://colab.research.google.com/]



## 全体の流れ

1. 学習データを用意する。合わせてテストデータも用意しましょう。
1. モデルを構築する
1. 学習をする/定期的にevaluationする
1. テストする



## データセットの用意


まずはともあれ画像データを用意する必要があります。ChainerCVには大量の画像データセットを1行で取得できる簡単な仕組みを備えています。

```
import numpy as np
import random

from chainercv.datasets import OnlineProductsDataset
from chainercv.datasets import online_products_super_label_names as label_names
from chainercv.transforms import resize
from chainercv.utils import tile_images
from chainercv.visualizations import vis_image

dataset = OnlineProductsDataset()
images = []
for c in range(5):
  i = random.randint(0, dataset.__len__())
  _data = dataset.get_example(i)
  _image = resize(_data[0], (200,200))
  _label = _data[2]
  print("label:{},{}".format(_label, label_names[_label]))
  images.append(_image)

tile_image = tile_images(np.array(images), 5)
vis_image(tile_image)

```

最初はデータセットの取得に時間がかかりますが、取得出来たデータのサンプルが確認できるようになりました。サンプル画像がその画像を表しているラベルとともに表示されましたね。


![データセットのサンプル](src/images/chug_03_dataset_sample.png)

このデータセットはStanford Online Products datasetといって一般に公開されているデータであり、Ebayの12のの分類がされたデータです。


トレーニングデータとテストデータの両方を見せて、画像がかぶっていないことを確認させる



## 画像変換クラスの作成

```
class TrainTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        img = random_sized_crop(img)
        img = resize(img, (224, 224))
        img = random_flip(img, x_random=True)
        img -= self.mean
        return img, label
      
```

TrainTransformの意義を説明する。random_size_crop, resize, random_flipがあることの効果を論文を引用して説明する。



```
class ValTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        img = scale(img, 256)
        img = center_crop(img, (224, 224))
        img -= self.mean
        return img, label
```

ValTransformの意義を説明する。TrainTransoformと比較して、center_cropである理由を説明する。



## モデルの作成

```
from chainer.links import Classifier
from chainercv.links import ResNet50
from chainercv.links.model.resnet import Bottleneck

lr=0.1

extractor = ResNet50(n_class=len(label_names), **{'arch': 'fb'})
extractor.pick = 'fc6'

model = Classifier(extractor)

for l in model.links():
    if isinstance(l, Bottleneck):
        l.conv3.bn.gamma.data[:] = 0
```

Chainerでのモデルの作成はResNetであってもこんなにかんたんなんだよ。ということを説明する。できれば、自前でモデルを作成する方法も説明したいが、ここでは不要かな。



## 学習

Trainerを使った学習のログを示す




## テスト

テストする。


## まとめ

まとめする。

