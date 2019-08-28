# Chainer を AMD の GPU で動かそう

@<icon>{cheiko} 「ありえない・・・」

@<icon>{cheita} 「どうした」

@<icon>{cheiko} 「ディープラーニングしたいから GPU がたくさん欲しいなってつぶやいていたら自称ちぇい子ファンという 30 代男性（無職）から GPU が送られてきたの」

@<icon>{cheita} 「すでに事案という雰囲気がただよっているが大丈夫か？」

@<icon>{cheiko} 「でもその GPU が全部 AMD の GPU だったの・・・これじゃディープラーニングに使えないわ！GPU は NVIDIA じゃないと意味がないじゃない！」

@<icon>{yousei} 「大丈夫！実は Chainer は AMD の GPU 対応も進めているんだよ。」

@<icon>{cheiko} 「そもそもあの人プロフィール欄に "ビットコイン" とか書いてあったから怪しいと思ってたの・・・きっとマイニングに使ってた GPU が要らなくなったから送りつけてきただけなんだわ・・・！」

@<icon>{yousei} 「話聞いてる？」

@<icon>{cheita} 「前後の章とキャラの一貫性がなくなるからその辺にしとけよ」

@<icon>{cheiko} 「こうなったら本来の用途で使ってやる！ディープラーニングなんてやめて PUBG やるわよ！」

@<icon>{yousei} @<icon>{cheita} 「「待って！！この本の趣旨が変わっちゃう！！」」

## 準備

@<icon>{yousei} 「Chainer を AMD の GPU で動かすには、まず Chainer で標準の GPU バックエンドになっている CuPy というライブラリを AMD の GPU で動かせるようにする必要があるよ。」

@<icon>{cheita} 「といっても CuPy の公式ページをみても AMD GPU に関する記述は見当たらないな・・・どうするんだ？」

@<icon>{yousei} 「まだ CuPy の AMD GPU 対応版は開発中で、CuPy の作者 okuta 氏のプルリクエストからコードを引っ張ってきて自分でビルドする必要があるよ。」

@<icon>{cheita} 「なんだか大変そうだな・・・」

@<icon>{yousei} 「大丈夫！まずは AMD からリリースされている ROCm Platform を Ubuntu が入った PC 上にセットアップする必要があるけど、以下の手順でコマンドを実行していくだけだ。」

### ROCm インストールの手順

（以下の手順は ROCm 公式リポジトリの README に記載されている手順：https://github.com/RadeonOpenCompute/ROCm#ubuntu-support---installing-from-a-debian-repository と同一となります。）

以下では、Ubuntu 18.04 を前提としています。

まずシステムを最新の状態に更新しましょう。
ついでに `libnuma-dev` をインストールしておきます。

```bash
sudo apt update
sudo apt dist-upgrade
sudo apt install libnuma-dev
```

アップグレードが実行された場合、一度再起動を行っておきましょう。

```bash
sudo reboot
```

次に ROCm のリポジトリを追加します。

```bash
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
```

ROCm のリポジトリが追加できたら、`rocm-dkms` というメタパッケージをインストールします。

```bash
sudo apt update
sudo apt install rocm-dkms
```

最後に、ROCm を使用したいユーザを `video` グループに追加します。
以下のコマンドでは、現在のログインユーザを追加しています。

```bash
sudo usermod -a -G video $LOGNAME
```

以上で、基本的な環境構築は完了です。

@<icon>{yousei} 「次はいよいよ CuPy のビルドだ！」

### AMD GPU 対応 CuPy のビルド

まずは以下の必要パッケージを追加インストールします。

```bash
sudo apt install hipblas hipsparse rocrand
```

次に、いくつかの環境変数を設定します。

```bash
export HCC_AMDGPU_TARGET=gfx900
export ROCM_HOME=/opt/rocm
export CUPY_INSTALL_USE_HIP=1
export PATH=$ROCM_HOME/bin:$PATH
export CUDA_PATH=/opt/rocm
```

このとき、`HCC_AMDGPU_TARGET` に与える値は、使っている AMD GPU によって変更する必要があります。
ROCm がサポートしている GPU や CPU の一覧は、こちらから調べることができます：https://rocm.github.io/hardware.html

次に、CuPy のビルドに必要な Python パッケージをインストールします。

```bash
pip3 install cython numpy
```

それでは、開発中の AMD GPU 対応 CuPy のブランチをクローンして、ビルドしましょう。

```bash
# ブランチのクローン
git clone https://github.com/okuta/cupy -b support-hip

cd cupy

# CuPy のビルドとインストール
CFLAGS="-I/opt/rocm/include -I/opt/rocm/hiprand/include -I/opt/rocm/rocrand/include" \
LDFLAGS="-L/opt/rocm/lib -L/opt/rocm/hiprand/lib -L/opt/rocm/rocrand/lib -L/opt/rocm/hip/lib" \
pip3 install -e . -vvv
```

エラーなく終了すれば、CuPy が無事 AMD GPU 上で動くようになっているはずです。
以下のコマンドで動作を確認してみましょう。

```bash
# cupy ディレクトリ以下で
cd examples/gemm
python3 sgemm.py
```

無事に以下のような結果が表示されれば、成功です。

```bash
m=1445 n=1012 k=1076
start benchmarking

=============================Result===============================
hand written kernel time 1.5921421766281127 ms
cuBLAS              time 0.6761380195617676 ms
```

### 環境構築がされた Docker イメージを使う

〜うまくいかない場合〜

@<icon>{cheita} 「（うーん、なぜかエラーが出る・・・）」

@<icon>{cheiko} 「私の GPU でディープラーニングできるようになったかしら？」

@<icon>{cheita} 「いや、なぜかうまくいかない・・・」

@<icon>{cheiko} 「じゃゲームしてるね！」

@<icon>{cheita} 「おい！」

@<icon>{yousei} 「仕方ないなあ。事前に動作確認ができている Docker イメージを作ったから、それを使ってみてよ。ただし、事前に `rocm-dkms` をホストOSにインストールしておく必要はあるよ。」

@<icon>{cheita} 「Docker イメージを使う場合でも、"ROCm インストールの手順" までは済ませておく必要があるってことだね。」

@<icon>{yousei} 「その通り！」

まず　Docker Hub から以下の Docker イメージを pull してきます。

```bash
docker pull mitmul/techbookfest7:cupy-amd
```

pull してきた Docker イメージを使って、CuPy の example を実行してみます。

```bash
docker run --rm -ti \
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
mitmul/techbookfest7:cupy-amd \
python3 cupy/examples/gemm/sgemm.py
```

以下のような出力が表示されれば、無事に動作しています。

```bash
m=1445 n=1012 k=1076
start benchmarking

=============================Result===============================
hand written kernel time 1.5921421766281127 ms
cuBLAS              time 0.6761380195617676 ms
```

## AMD GPU で Chainer の MNIST example を動かす

@<icon>{cheiko} 「ダダダダダダダダ！！！撃て撃てえええ！！」

@<icon>{cheita} 「おい！AMD GPU で Chainer を動かす準備ができたぞ！ゲームやめろ！！」

@<icon>{cheiko} 「ちょっと待ってなんか戦場に犬がいるわ。ウロチョロと邪魔臭いわね・・・」

@<icon>{yousei} 「ビクンッ」

@<icon>{cheita} 「いいから定番の MNIST example でも走らせてみようぜ。」

@<icon>{cheiko} 「あら。MNIST なら朝飯前よ。私にまかせて！」

AMD GPU に対応した CuPy がインストールされていれば、Chainer 側では特に特別なことをする必要はありません。
現時点の AMD GPU 対応 CuPy のブランチは CuPy v7.0.0b2 をベースにしているため、Chainer インストール時にこのバージョンだけ一応合わせておきます。

```bash
pip3 install chainer==7.0.0b2 matplotlib
```

Chainer のリポジトリをクローンします。

```bash
git clone https://github.com/chainer/chainer -b v7.0.0b2 --depth=1
```

MNIST example を `--gpu 0` オプション付きで実行してみましょう。

```bash
python3 chainer/examples/mnist/train_mnist.py -g 0
```

以下のような出力が表示されれば、成功です。
無事に AMD GPU を使って、Chainer で書かれたニューラルネットワークを MNIST データセットを使って訓練することができています。

```bash
Device: @cupy:0
# unit: 1000
# Minibatch-size: 100
# epoch: 20

Downloading from http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz...
Downloading from http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz...
Downloading from http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz...
Downloading from http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz...

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           0.191636    0.104016              0.941749       0.9677                    14.0415       
2           0.0725324   0.0809784             0.977365       0.9736                    17.9653       
3           0.0483136   0.07426               0.984282       0.9781                    21.9157       
4           0.0356667   0.0665017             0.988582       0.9802                    25.8826       
5           0.029242    0.0783757             0.990431       0.976                     29.8337       
6           0.0240914   0.0915075             0.991882       0.976                     33.7948       
7           0.0202024   0.070404              0.993548       0.9807                    37.726        
8           0.0171451   0.0737094             0.994632       0.9824                    41.7049       
9           0.0175979   0.0860599             0.994165       0.9818                    45.6674       
10          0.0174146   0.08188               0.994615       0.9804                    49.6225       
11          0.013079    0.078925              0.995815       0.9824                    53.5648       
12          0.0110601   0.0960848             0.996232       0.9828                    57.5095       
13          0.0115849   0.0925806             0.996599       0.9822                    61.4513       
14          0.014988    0.0839504             0.995399       0.9818                    65.4373       
15          0.0106209   0.0794084             0.996682       0.9832                    69.4073       
16          0.00842235  0.10651               0.997649       0.9803                    73.3788       
17          0.00726848  0.102539              0.997733       0.9815                    77.3515       
18          0.0119389   0.0910986             0.996699       0.9838                    81.3353       
19          0.0109424   0.0907164             0.997066       0.983                     85.3474       
20          0.00757545  0.107515              0.997699       0.9802                    89.3063   
```

@<icon>{cheiko} 「動いたわね。これは CPU と比べて速くなっているのかしら？」

@<icon>{cheita} 「手元の i7-6800K CPU @ 3.40 GHz と Radeon RX Vega 10 を比較すると、後者の方が 5 倍程度高速に訓練を回せているみたいだね。」

@<icon>{cheiko} 「やっぱりエヌビｄぃ」

@<icon>{cheita} 「（さえぎって）色々なハードウェアがサポートされるのはいいことだね！！」

@<icon>{yousei} 「もうじき某ブランチもマージされて、CuPy は公式に AMD GPU をサポートする予定だよ。」

@<icon>{cheiko} 「それはドン勝ね！」

## 最後に

* 今回使用した Docker イメージをビルドするための Dockerfile は、以下のリポジトリで配布しています。：https://github.com/chainer-community/techbookfest7/tree/master/src/docker/cupy-amd/Dockerfile
* Vega 10 と CPU を用いた実験結果のログも、上記リポジトリにある以下のファイルにまとめられていますので、興味のある方は参照してください。：https://github.com/chainer-community/techbookfest7/tree/master/src/docker/cupy-amd/README.md
* 現在、NVIDIA GPU 用の CuPy が Thrust を使っているもの（sort や argsort など）はサポートされていないため、例えば ChainerCV の物体検出の example などは動きません（non-maximum suppression (NMS) で cupy.argsort が必要なため）。hipThrust (https://github.com/ROCmSoftwarePlatform/Thrust) というものも開発されているようですが、CuPy での対応は未定です。sort を行う CUDA カーネルを新たに書いて、それを hip でコンパイルしたものを使うように CuPy 内部で AMD GPU 環境下で sort / argsort が呼ばれた際の動作を変更する（現在は Thrust が無い、という RuntimeError が出る）修正を行えば、速度は遅いものの sort / argsort が使えるようにはなると思います。CuPy はオープンソースですので、皆で機能を追加していきましょう！

#### 注意

著者は PUBG をプレイしたことが一度もないため、誤った認識に基づく記述が含まれる可能性がありますが、ご容赦いただけましたら幸いです。
