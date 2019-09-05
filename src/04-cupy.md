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

`/opt/rocm` 以下に様々なツールなども含め必要なものがインストールされるので、次回ログイン時から必要なパスなどが通っている状態になるように、以下のような環境変数の設定を `~/.bashrc` に追加しておきます。

```bash
echo 'export ROCM_HOME=/opt/rocm' >> ~/.bashrc
echo 'export PATH=$ROCM_HOME/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_PATH=/opt/rocm' >> ~/.bashrc
```

最後に、ROCm を使用したいユーザを `video` グループに追加します。
以下のコマンドでは、現在のログインユーザを追加しています。

```bash
sudo usermod -a -G video $LOGNAME
```

一度ログアウトして、ログインし直してください。
そして以下のコマンドを実行してみましょう。

```bash
rocm-smi
```

ROCm の環境構築がうまくいっていれば、以下のような（数値等は GPU の種類によって異なります）出力が表示されます。

```bash
 ========================ROCm System Management Interface========================
================================================================================
GPU  Temp   AvgPwr  SCLK     MCLK    Fan    Perf  PwrCap  VRAM%  GPU%  
0    50.0c  10.0W   1269Mhz  167Mhz  14.9%  auto  220.0W    0%   0%    
================================================================================
==============================End of ROCm SMI Log ==============================
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
export CUPY_INSTALL_USE_HIP=1
```

このとき、`HCC_AMDGPU_TARGET` に与える値は、使っている AMD GPU によって変更する必要があります。
ROCm がサポートしている GPU や CPU の一覧は、こちらから調べることができます：https://rocm.github.io/hardware.html

次に、CuPy のビルドに必要な Python パッケージをインストールします。
ここからは簡単のため、システムに `apt` でインストールされている `python3` を用います。
もし `pip3` がインストールされていなければ、まず

```bash
sudo apt install python3-pip
```

として `pip3` コマンドをインストールしてください。
`pip3` が準備できたら、`cython` と `numpy` パッケージをインストールします。

```bash
pip3 install cython numpy
```

それでは、開発中の AMD GPU 対応 CuPy のブランチをクローンして、ビルドしましょう。

```bash
# ブランチのクローン
git clone https://github.com/okuta/cupy -b support-hip --depth=1

cd cupy

# CuPy のビルドとインストール
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
1           0.192037    0.0958304             0.9416         0.9711                    136.969       
2           0.0754357   0.080027              0.976299       0.9745                    141.146       
3           0.0480771   0.0853976             0.984765       0.9744                    145.713       

(...途中省略...)

18          0.00856742  0.114604              0.997416       0.9817                    209.29        
19          0.00997513  0.105292              0.997283       0.9817                    213.263       
20          0.00816262  0.127226              0.997632       0.9807                    217.057       
```

@<icon>{cheiko} 「動いたわね。これは CPU と比べて速くなっているのかしら？」

@<icon>{cheita} 「オプションを指定せずに CPU を使って訓練を行ったとき（Intel Core i3-7100 CPU @ 3.90GHz 使用）とこの AMD GPU 使用時（Radeon RX Vega 10）を比較すると、後者の方が 9 倍程度高速に訓練を回せているみたいだね。CPU を使ったときの 1 エポックにかかる時間の平均は 35.4 秒程度で、AMD GPU を使った場合は 3.95 秒程度だった。ちなみに、Google Colab 上で無料で使える NVIDIA T4 GPU を使った場合は、3.33 秒程度だった。」

@<icon>{cheiko} 「ってことはやっぱりエヌビｄぃ」

@<icon>{cheita} 「（さえぎって）GPU の性能や CPU によっても訓練スクリプトの動作時の性能は変わってくるから、この数字を単純比較はできないよ！ともかく、色々なハードウェアがサポートされるのはいいことだね！！」

@<icon>{yousei} 「もうじき某ブランチもマージされて、CuPy は公式に AMD GPU をサポートする予定だよ。」

@<icon>{cheiko} 「それはドン勝ね！」

## 最後に

* 今回の実験や環境構築などは全て GPU Eater (https://www.gpueater.com/) の AMD GPU インスタンス `a1.vegafe` 上で行いました。
* 今回使用した Docker イメージをビルドするための Dockerfile は、以下のリポジトリで配布しています。：https://github.com/chainer-community/techbookfest7/tree/master/src/docker/cupy-amd/Dockerfile
* Vega 10 と CPU を用いた実験結果のログも、上記リポジトリにある以下のファイルにまとめられていますので、興味のある方は参照してください。：https://github.com/chainer-community/techbookfest7/tree/master/src/docker/cupy-amd/README.md
* 現在、NVIDIA GPU 用の（通常の）CuPy が内部で Thrust を使っている関数（`sort` や `argsort` など）はサポートされていないため、例えば ChainerCV の物体検出の example などは動きません（non-maximum suppression (NMS) で `cupy.argsort` が必要なため）。hipThrust (https://github.com/ROCmSoftwarePlatform/Thrust) というものも開発されているようですが、CuPy での対応は未定です。sort を行う CUDA カーネルを新たに書いて、それを hip でコンパイルしたものを使うように CuPy 内部で AMD GPU 環境下で `sort` / `argsort` が呼ばれた際の動作を変更する（現在は Thrust が無い、という RuntimeError が出る）修正を行えば、速度は遅いものの `sort` / `argsort` が使えるようにはなると思います。CuPy はオープンソースですので、皆で機能を追加していきましょう！

#### 注意

著者は PUBG をプレイしたことが一度もないため、誤った認識に基づく記述が含まれる可能性がありますが、ご容赦いただけましたら幸いです。
