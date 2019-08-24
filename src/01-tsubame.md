# TSUBAME3.0でChainerMN

@<icon>{yousei} 「せっかく大岡山に来たことだし、まずはTSUBAMEでChainerMNをしてみようか！」

@<icon>{yousei} 「そもそもTSUBAMEっていうのは、東京工業大学に設置された大規模クラスター型スーパーコンピュータのことなんだ。」

@<icon>{cheiko}「そうなのね！スパコンでDeep Learningできるんだ。」

@<icon>{yousei} 「スパコンといってもアクセラレータ自体は市販のGPUと変わらないから、普通のGPUクラスタマシンと考えても問題ないんだよ。」

@<icon>{cheita} 「TSUBAME3.0は、1ノードあたりNVIDIA P100を4枚、計540ノードで国内有数のスパコン！単精度で24.3PFlopsで~~」

@<icon>{cheiko}「・・・(ちぇい太君の知識自慢が始まってしまった。)」

## ChainerMNとは
ChainerMNは、Chainerを用いた学習を分散処理により高速化する機能です。
Chainer Multi Nodeを意味しています。
柔軟で直感的に利用できる Chainer の利便性をそのままに、学習時間を大幅に短縮できます。
1ノード内の複数のGPUを活用することも、複数のノードを活用することもできます。
最新のChainerでは、Chainerと一緒にインストールされますので、必要な環境設定を行えばすぐに使うことができます。

## TSUBAME3.0でChainerMN
### TSUBAME3.0へログイン
一般的なクラウド環境のインスタンスのようにログインすることが可能です。詳しくは https://helpdesk.t3.gsic.titech.ac.jp/manuals/handbook.ja/start/ を参照ください。

### Chainerのインストール

@<icon>{yousei} 「まずは、Chainerのインストールをしようか。ログインノードに入って以下のコマンドを打ってみよう。」

```
xxxxx@login0:~> GROUP="自分のグループ名"
xxxxx@login0:~> qrsh -g $GROUP -l s_gpu=1 -l h_rt=1:00:00
xxxxx@r8i6n8:~> . /etc/profile.d/modules.sh
xxxxx@r8i6n8:~> module load python/3.6.5
xxxxx@r8i6n8:~> module load cuda/9.2.148
xxxxx@r8i6n8:~> module load openmpi/2.1.2-opa10.9
xxxxx@r8i6n8:~> pip install --user mpi4py cupy-cuda92==6.2.0 chainer==6.2.0 
xxxxx@r8i6n8:~> python -c 'import chainer; chainer.print_runtime_info()'
Platform: Linux-4.4.121-92.85-default-x86_64-with-SuSE-12-x86_64
Chainer: 6.2.0
NumPy: 1.17.0
CuPy:
  CuPy Version          : 6.2.0
  CUDA Root             : /apps/t3/sles12sp2/cuda/9.2.148
  CUDA Build Version    : 9020
  CUDA Driver Version   : 10000
  CUDA Runtime Version  : 9020
  cuDNN Build Version   : 7402
  cuDNN Version         : 7402
  NCCL Build Version    : 2402
  NCCL Runtime Version  : 2402
iDeep: Not Available
```

@<icon>{cheiko}「何をやっているかよく分からないけど簡単ね！」

@<icon>{cheita} 「TSUBAME3.0で動作確認が取れているpython, cuda, openmpiをロードして、それに対応しているchainer, cupyをインストールしているんだよ。」

#### 注意

* 環境は、特段の理由がない限り、CUDA 9.2, Open MPI 2.1.2 が推奨されます。
* Chainerのバージョンは、6.2.0 を推奨します。また、明示的にインストールするバージョンを指定したほうが安全です
* CuPYは、`cupy-cuda92` というバイナリインストールを用いるのが推奨です。単に`cupy`と指定するとソースからのビルドになりますが、その場合、NCCLとCuDNNを自分でインストールしてした上でビルドする必要が出てきます。

### スクリプト(MNIST)の準備

@<icon>{yousei} 「まずは、分散環境で動かすサンプルの実行スクリプトとデータを用意しよう。」

//blankline

ChainerのMNISTサンプルは、初回実行時に`HOME`ディレクトリにMNISTデータをダウンロードします。
計算ノードからはインターネットにアクセスできないので、ログインノードでMNISTを一回実行してデータをダウンロードさせる必要があります。
この実行はChainerMNである必要はないので、通常のMNISTのサンプルコードを使ってデータのダウンロードをしましょう。

#### 注意

* まちがえてmasterブランチのtrain_mnist.pyをダウンロードするとエラーで実行できないです。
* なお、手動でデータをコピーしても大丈夫です。

```
# NOTE: env.shの実行を忘れないように
$ wget https://raw.githubusercontent.com/chainer/chainer/v5.1.0/\
examples/mnist/train_mnist.py -O train_mnist_single.py
$ python train_mnist_single.py -e 1
```

@<icon>{yousei} 「次に、ChainerMN用の`train_mnist.py`をダウンロードしよう。」

@<icon>{yousei} 「くれぐれも、masterブランチのtrain_mnist.pyをダウンロードしないように。バージョンが対応している必要があるんだ。」

```
$ wget https://raw.githubusercontent.com/chainer/chainer/v5.1.0/\
examples/chainermn/mnist/train_mnist.py
```

### ジョブスクリプトの準備

@<icon>{yousei} 「ChainerMNプログラムを分散環境で動かすには、動かしたいプログラムの他に2つのファイルが必要なんだ。」

 * 実行スクリプト `train_mnist.py`
 * 動かしたいアプリケーションを起動するジョブスクリプト（ここでは例として以下の`job_mnist.sh`)
 * 補助ファイル `run.sh` 

job_mnist.sh

```
#!/bin/sh
#$ -cwd

. /etc/profile.d/modules.sh
module load python/3.6.5
module load cuda/9.2.148
module load openmpi/2.1.2-opa10.9

# ジョブのはじめに、常にバージョン情報や日時等の情報を出力しておくのは良い習慣です
date
python -c 'import chainer; chainer.print_runtime_info()'
cat $PE_HOSTFILE

# OMP_NUM_THREADSは、OpenCVをChainerから（直接・間接的に）使う場合に性能に影響します。
# 特段の理由がなければ 1 にしておくのが無難です
export OMP_NUM_THREADS=1

# PPN: ノードあたりのプロセス数です。
# 使うキューによって変動します。Chainerを使う場合は、基本的にホストあたりのGPU数にします
# s_gpu のとき: 1
# f_node のとき: 4
export PPN=1
echo PPN=$PPN

# 総プロセス数を計算で求めます
NP=$(expr $PPN \* $NHOSTS)
echo NHOSTS=$NHOSTS
echo NP=$NP

# MNISTを実行します
# 1行目は、基本的には変更する必要はなく、2行目を自分のアプリケーションに従って変更します。
# コツ：
#    * ChainerMNのコミュニケーターは、 pure_nccl を利用しましょう。
#    * データの出力先となるディレクトリ名には、$JOB_ID をつけておくのがおすすめです
mpiexec -npernode $PPN -n $NP -x LD_LIBRARY_PATH $HOME/run.sh \
  python train_mnist.py -g --communicator='pure_nccl' -e 2 -o result.$JOB_ID
```

run.sh

```
#!/bin/sh
# 分散環境においてPythonスクリプトを実行するための補助ファイルです。
# これを $HOME/run.sh という名前で保存します
# （保存先とファイル名は任意ですが、変更する場合は job_mnist.sh の中のファイル名も変更してください）

. /etc/profile.d/modules.sh

module load python/3.6.5
module load cuda/9.2.148
module load openmpi/2.1.2-opa10.9

"$@"
```

@<icon>{yousei} 「ジョブスクリプトの内容は、走らせたいジョブの処理の内容に従って細かく変更してね。」

@<icon>{yousei} 「例えば、1ノードあたり4GPUを使いたい場合はf_nodeを指定する必要があったりするんだ。」

次に、テストとしてMNISTを並列学習するジョブを投入してみます。以下の `job_mnist.sh` と `run.sh` の2つのファイルを用意して、実行権限を付与します

```
xxxxx@login0:~> vi job_mnist.sh
xxxxx@login0:~> chmod +x job_mnist.sh

xxxxx@login0:~> vi run.sh
xxxxx@login0:~> chmod +x run.sh
```

ジョブを投入します。

```
xxxxx@login0:~> qrsh -g $GROUP -l s_gpu=4 -l h_rt=1:00:00 ./job_mnist.sh
Sat Aug  3 21:37:25 JST 2019
Platform: Linux-4.4.121-92.85-default-x86_64-with-SuSE-12-x86_64
Chainer: 6.2.0
NumPy: 1.17.0
CuPy:
  CuPy Version          : 6.2.0
  CUDA Root             : /apps/t3/sles12sp2/cuda/9.2.148
  CUDA Build Version    : 9020
  CUDA Driver Version   : 10000
  CUDA Runtime Version  : 9020
  cuDNN Build Version   : 7402
  cuDNN Version         : 7402
  NCCL Build Version    : 2402
  NCCL Runtime Version  : 2402
iDeep: Not Available
r1i7n6 2 all.q@r1i7n6 <NULL>
r2i3n1 2 all.q@r2i3n1 <NULL>
r8i6n8 2 all.q@r8i6n8 <NULL>
r2i4n8 2 all.q@r2i4n8 <NULL>
==========================================
Num process (COMM_WORLD): 4
Using GPUs
Using pure_nccl communicator
Num unit: 1000
Num Minibatch-size: 100
Num epoch: 2
==========================================
--------------------------------------------------------------------------
A process has executed an operation involving a call to the
"fork()" system call to create a child process.  Open MPI is currently
operating in a condition that could result in memory corruption or
other system errors; your job may hang, crash, or produce silent
data corruption.  The use of fork() (or system() or other calls that
create child processes) is strongly discouraged.

The process that invoked fork was:

  Local host:          [[19872,1],0] (PID 21643)

If you are *absolutely sure* that your application will successfully
and correctly survive a call to fork(), you may disable this warning
by setting the mpi_warn_on_fork MCA parameter to 0.
--------------------------------------------------------------------------
epoch main/loss   validation/main/loss  main/accuracy  validation/main/accuracy
1     0.289679    0.118232              0.915267       0.9628
2     0.0913869   0.0768252             0.972          0.9755
```

出力された結果を見て、正しく実行されたかどうか確認しましょう。

 * 冒頭に `chainer.print_runtime_info()` の結果が出力されているので、ライブラリが正しく読み込まれていることを確認します
 * `-l s_gpu=4` という指定をしましたので、合計4プロセスで実行されます。 出力の`Num process (COMM_WORLD): 4` と一致していることを確認します。
 * コミュニケーターとして `pure_nccl communicator` が使われていることを確認します
 * MNISTが正しく学習できていることを確認します。（注： なお、MNISTは計算負荷が軽いので、複数GPU実行ではむしろ遅くなるケースが多いです）

以上でChainerMNプログラムを分散学習できました。

## TSUBAME3.0の利用申請
そもそもTSUBAMEを使うためには利用申請が必要になります。
東工大の学生や教職員の方しか使えないイメージがありますが、
学外の方も申請すれば使えるようです。

詳しくは、https://www.t3.gsic.titech.ac.jp/getting-account に説明がありますのでご参照ください。

## おわりに
敷居が高いように見えるスパコンも実はこんなに簡単に使うことができます。
また、第1回 ディープラーニング分散学習ハッカソン(http://gpu-computing.gsic.titech.ac.jp/node/100)というイベントなども開催していたりします。ぜひ興味がある方は、まずはハッカソンにでも参加してみたらどうでしょうか。

### 参考文献
* 第1回 ディープラーニング分散学習ハッカソンの当日資料(https://bit.ly/t3-optuna)
