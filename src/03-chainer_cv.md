# しったかChainerCV～ResNetで画像識別してみよう～


@<icon>{yousei} 「前回はChainerCVを使って画像処理の基礎についてお話したよね。覚えている？」

　

@<icon>{cheiko} 「もちろんよ。画像のチャネル構成や、画像変形、水増し、データセットについて学んだし復習もしたから大丈夫よ。」

　

@<icon>{yousei} 「じゃあ今回は、ちょっと進めてResNetモデルを使って本格的にDeepLearningで画像分類をしてみよう。Chainerを使うと簡単に実装できることがわかるよ。」

　

@<icon>{cheiko} 「れずねっと？」

　

@<icon>{cheita} 「ResNetだよ。DeepLearning界隈で有名なImageNetの分類問題のコンペであるILSVRC 2015でエラー率3.57%となり、1位を獲得したモデルさ。正式にはResidualNetといい、残差を・・・」

　

@<icon>{cheiko} 「・・・（またはじまったわ。この子）」

　

@<icon>{yousei} 「よく知っているね。今日はその高精度なモデルResNetであっても実装は簡単だということを見せてあげよう。」

　
　

先生の言う通り、DeepLearningで画像分類と言ってもChainerを使えば難しくはありません。手順どおりに進めていくだけでできちゃいます。なにはともあれChainerCVをインストールしてはじめよう！


```
!pip install chainercv==0.12
```

#### 注意

* この章ではGoogleColabratory@<fn>{fn01}上で実行することを想定しています。
* GoogleColabratoryではGPUを使うように設定してください。メニューの[ランタイム]-[ランタイムのタイプを変更]で[ハードウェアアクセラレータ]項目に[GPU]を設定してください。
* 執筆時点でGoogleColabratoryはChainer 5.4.0がインストールされているため合わせてChainerCVは0.12を指定しています。ChainerとChainerCVとの組み合わせはhttps://github.com/chainer/chainercv を参照してください。

//footnote[fn01][https://colab.research.google.com/]



## データセットの用意


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

これだけで学習用画像データセットとテスト用画像データセットが取得できます。さて、どれくらいのデータ量なのか確認してみましょう。

```
print("dataset train size:{}".format(len(dataset_train)))
print("dataset test size:{}".format(len(dataset_test)))
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

この画像データセットはStanford大学が公開しているOnline Products dataset@<fn>{fn02}です。オンライン販売のEbayに登録されている商品を12の分類(bicycle, cabinet, chainer, coffe_maker, fan, kettle, lamp, mug, sofa, stapler, table, toaster)したデータです。

//footnote[fn02][http://cvgl.stanford.edu/projects/lifted_struct/]


画像データセットには学習用データセット(train)とテスト用データセット(test)があります。これらは、学習用データセットを使いモデルを構築して、テスト用データセットを使い構築したモデルを評価する。というように使います。そのため、学習データとテスト用データで画像が混ざってはいけません。それではカンニングになってしまいますからね。




## 画像変換クラスの作成

次に画像変換クラスを作成しておきます。

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

TrainTransformの意義を説明する。random_size_crop, resize, random_flipがあることの効果を論文を引用して説明する。
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

