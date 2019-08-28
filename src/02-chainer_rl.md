# しったかDQN〜ChainerRLを使う前に〜 @ikeyasu

@<icon>{yousei} 「ChainerRLで簡単に強化学習ができるね！ブロック崩しをやってみようよ」

@<icon>{cheiko}「そうね！ChainerRLなら簡単！」

@<icon>{cheita} 「そもそも、ChainerRLで使えるアルゴリズムをちゃんと知っているの？それも知らずにサンプルを動かして分かった気になってない？」

@<icon>{cheita} 「そもそも、強化学習の基本は分かっているみたいだけど、個別のアルゴリズムについても知らないと、どう使って良いのか分からないよ。ブロック崩しなら、まずは、DQNでやると思うけど、どんなアルゴリズムか知ってる？」

@<icon>{cheiko}「・・・(ちぇい太君の知識自慢が始まってしまった。)」

## ChainerRLとは

ChainerRLとは、強化学習を簡単にするものです。chugの「ちぇい子と学ぶ Chainer 入門」をご参照下さい。
前回はChainerRLのサンプルを元に、簡単に使い方をせつめいしました。本稿では、ChainerRLのDQNのサンプルが
分かるよう、そのアルゴリズムの説明をしたいと思います。

※ 以下の記事は、著者が書いたQiitaの記事、[趣味の強化学習](https://qiita.com/ikeyasu/items/67dcddce088849078b85)を一部再構成しています。


## ChainerRLのサポートするアルゴリズム

ChainerRLのサポートするアルゴリズムは、2019/08/25現在、以下の通りです。

| Algorithm | Discrete Action | Continous Action | Recurrent Model | CPU Async Training |
|:----------|:---------------:|:----------------:|:---------------:|:------------------:|
| DQN  | ✓ | ✓ (NAF) | ✓ | x |
| Categorical DQN | ✓ | x | ✓ | x |
| Rainbow | ✓ | x | ✓ | x |
| IQN | ✓ | x | ✓ | x |
| DDPG | x | ✓ | ✓ | x |
| A3C  | ✓ | ✓ | ✓ | ✓ |
| ACER | ✓ | ✓ | ✓ | ✓ |
| NSQ  | ✓ | ✓ (NAF) | ✓ | ✓ |
| PCL | ✓ | ✓ | ✓ | ✓ |
| PPO  | ✓ | ✓ | ✓ | x |
| TRPO | ✓ | ✓ | x | x |
| TD3 | x | ✓ | x | x |
| SAC | x | ✓ | x | x |

横軸は、以下の通りの意味になります。

* "Discrete Action" = 離散アクションに対応。ゲームのように、ボタンを押して操作するものはこれに該当
* "Continous Action" = 連続アクションに対応。自動運転、ロボット制御など、行動が連続値で表されるものに該当
* "Recurrent Model" = 入力の部分をRecurrentにする事ができる。
* "CPU Async Training" = 非同期で複数のエージェントを動作させて学習できる。

このように、アルゴリズム毎にRecurrentや非同期の学習などをサポートしているのは、ChainerRLの特徴の一つです。様々なアルゴリズムを試すときに、これらを切り替えて試すことができます。

縦軸がアルゴリズムの名前で、主要なものはサポートされている状況です。本当に主要なものか？の参考までに、強化学習のフレームワークでおそらく一番有名な、Coachでサポートされているものと比較します。

| Coachでサポートされているアルゴリズム | ChainerRLでサポートされているか
|:----------|:---------------:|
| Deep Q Network (DQN) | ✓ |
| Double Deep Q Network (DDQN)  | ✓ |
| Dueling Q Network | ✓ |
| Mixed Monte Carlo (MMC) | |
| Persistent Advantage Learning (PAL) | ✓ |
| Categorical Deep Q Network (C51) | ✓ |
| Quantile Regression Deep Q Network (QR-DQN) |  |
| N-Step Q Learning | ✓ |
| Neural Episodic Control (NEC) |  |
| Normalized Advantage Functions (NAF)  | ✓ |
| Rainbow | ✓ |
| Asynchronous Advantage Actor-Critic (A3C) | ✓ |
| Deep Deterministic Policy Gradients (DDPG) | ✓ |
| Proximal Policy Optimization (PPO) | ✓ |
| Clipped Proximal Policy Optimization (CPPO) | |
| Generalized Advantage Estimation (GAE) | ✓ |
| Sample Efficient Actor-Critic with Experience Replay (ACER) | ✓ |
| Soft Actor-Critic (SAC) | ✓ |
| Twin Delayed Deep Deterministic Policy Gradient (TD3) | ✓ |

上記のように、大多数がサポートされています。名前のよくしれているアルゴリズムで非サポートなのは、NECぐらいですね。
ここからも、ChainerRLは、著名な強化学習アルゴリズムに既に対応していることがわかります。

## しったかDQN

DQN、、というと別のものを思い浮かべてしまう方は、、強化学習を勉強する方にはあまりいないでしょうね。DQNというと、Deep Q Network の事です。これは、Deep Learningを使って、Atariのゲームのスコアを著しく向上し、まさに深層強化学習の世界を切り開いたアルゴリズムです。
このアルゴリズムは理解しやすいため、本記事では入門として、DQNを取り上げます。

## アルゴリズムの分類

と書いてすぐになんですが、最初に強化学習アルゴリズムの分類をしておきます。

* 価値ベース（Value Optimization）
* 方策ベース（Policy Optimization）

しったからしく、ざっくり言うと、価値ベースは状態評価を元に学習をする手法、方策ベースは方策（戦略）を元に学習する手法です。
上記に上げたアルゴリズムを分類すると以下のようになります。

* 価値ベース
  * DQN
  * NEC
  * N-Step
  * PAL
  * Rainbow
* 方策ベース
  * A3C
  * DDPG
  * PPO
  * SAC
  * ACER
  * TD3

※ [Cachのサイト](https://github.com/NervanaSystems/coach#supported-algorithms)より抜粋

上記のように、DQNは、価値ベースの手法の一つです。方策ベースとは違う点が多いので、DQNが全ての基本！と思うのでは無く、価値ベースの一つの手法という位置づけで捉えておいてください。

## DQNとは

2013年のNIPS(現在のNeurIPS)に出された[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
および、科学雑誌Natureに2015年に掲載された[Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)の2つかいわゆるDQNの論文になります。
この2つは微妙にネットワーク構成が異なり、NIPS版、Nature版とそれぞれ呼ばれたりします。

## 基本の復習

強化学習をしているとでてくるおなじみの図です。

![強化学習の要素](src/images/chainerrl_fig1.png)

* 状態@<m>{s} (state) : 環境から得られる現在の状態。
* 行動@<m>{a} (action)：エージェントが与える行動。環境の上智娃に変化を与える。
* 報酬@<m>{r}(reward)：環境から得られた報酬

強化学習の基本として、報酬の最大化について理解する必要があります。
上記のようなループで、単に報酬が多いものを選ぶように繰り返したとしても、エージェントは知的な行動をするわけではありません。
短期的な報酬ではなく、長期的な報酬を最大化することが大切です。

これはつまり、報酬の総和を最大化するという事です。それを式で表すと以下のようになります。

//texequation{
R _t = r _{t} + \gamma r _{t+1} + \gamma ^2 r _{t+2} + \gamma ^3 r _{t+3} + ...
//}

この報酬の総和@<m>{R _t}は、収益と言います。
ここまで述べてきたとおり、報酬は、@<m>{r}です。
@<m>{t} は現在時刻で、@<m>{r _t}は現時点で、@<m>{r _{t + 1\\}}は1ステップ分、未来の報酬になります。
@<m>{\gamma}(ガンマ)は、割引率といって、未来の報酬をどれだけ割り引くかです。つまり、直近の報酬を優先するか、将来の報酬を優先するかを定義します。
例えば、0.1 だと将来の報酬はほとんど無視され、0.99だと将来の報酬は重視されて、よほど遠くならないと無視されません。

これは、ハイパーパラメーターで、、論文や公開されているソースを見ると、0.995 など1に近い値が使われることが多いようです。

## Q学習

それでは、本格的にDQNの話に入っていきます。先に述べたとおり、DQNは、Deep Q Networkです。つまり、"Q"という何かがあって、それの、Deep Network版なわけです。この"Q"は、Q学習（Q-leraning）です。

Q学習では、Q関数を更新して学習を進めます。Q関数は、状態sで行動aを行ったときの収益を推定する関数Qを求める関数で、以下のように表します。

//texequation{
Q(S _t, a _t)  \leftarrow Q(S _t, a _t)  + \alpha [r _{t+1} + \gamma \displaystyle \max _a Q(S _{t+1}, a) - Q(S _t, a _t)]
//}

@<m>{S _t} が時間tにおける状態、@<m>{a _t}は時間tにおける行動です。

@<m>{\alpha}はステップサイズ。0.1 など小さい値を用います。
@<m>{\gamma} は前述の割引率。
@<m>{\displaystyle \max _a Q(S _{t+1\\}, a)}は、将来の理想値を示しています。

上記の式を使った更新のアルゴリズムは以下のように書けます。

-----

* Q(s, a) を任意に初期化
* 各エピソードに対し繰り返し：
  * sを初期化
  * エピソードの各ステップに対して繰り返し：
      * Q関数と探索を使って、sでの行動aを選択する
      * 行動aを取り、r, s' を観測する
      * Q(s, a) ← Q(s, a) + @<m>{\alpha}[r + @<m>{\gamma max _{a'\\}} Q(s', a') - Q(s, a)]
      * s ← s';
  * sが終端状態ならば繰り返しを終了

※ [強化学習](https://amzn.to/2PgfGSD) P160から引用

-----

上記の実装例でもそうですが、Q関数を最もシンプルに実装すると、それは大きな辞書型のテーブルになります。

改めて、状態sで行動aを行ったときの収益を推定する関数ですので、以下のように、状態と行動に対応した収益が書かれたテーブルとなります。

|状態↓＼行動→  |前  |後ろ  |
|---|---|---|
|前にゴールがある  |6  |-1  |
|後ろにゴールがある |-1  |6  |

これは単純な問題、例えば簡単な迷路探索問題ならば確かに、テーブルで実装できます。ですが、より複雑な入力では組み合わせ数が爆発し、テーブルでは実装不可能です。

例えば、atari のブロック崩しは、64x64の画像が入力ですが、このときの状態の数は@<m>{256 ^{64\times 64\\}}(約@<m>{10 ^{9864\\}})です。とてもテーブルを作れる数ではありません。

そこで、このQ関数をDeep learningで近似する技術が近年発展しました。

## Q関数の近似

DQNでは、@<m>{Q(S, a)}をDeep neural networkで近似しています。特に、Atariのゲームは入力が加増なので、Convolutionを使って近似することになります。

前述の通り、Q関数は、状態sで行動aを行ったときの収益を推定する関数です。そのため、Q関数をそのまま近似すると、以下の通り、状態sと行動aを受け取って、Q値を返すようなニューラルネットワークになります。

![Convolutionを使ったQ関数の近似](src/images/chainerrl_fig2.png)

しかし、これですと、全てのActionのQ値を求めるために、上記のニューラルネットワークを行動の数だけ実行する必要があります。これは、とても遅いため、DQNでは以下のように、全ての行動のQ値をまとめて出すようにしています。

![Convolutionを使ったQ関数の近似](src/images/chainerrl_fig3.png)

## ChainerRLのQ関数

ChainerRLのAtariのDQNのサンプルの、[examples/atari/train_dqn_ale.py](https://github.com/chainer/chainerrl/blob/master/examples/atari/train_dqn_ale.py)を見てみましょう。
以下の、`parse_arch`でネットワークを作っています。

```
def parse_arch(arch, n_actions):
    if arch == 'nature':
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions),
            DiscreteActionValue)
    elif arch == 'doubledqn':
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions, nobias=True),
            SingleSharedBias(),
            DiscreteActionValue)
    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(),
            L.Linear(256, n_actions),
            DiscreteActionValue)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))
```

最初に書いたとおり、DQNには、NIPS版とNature版があります。
一つ目の、以下が、Nature版のネットワークで、これがQ関数を近似しています。

```
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions),
            DiscreteActionValue)
```

この中で、`NatureDQNHead`というクラスが出てきますが、これは[chainerrl.links.dqn_head](https://github.com/chainer/chainerrl/blob/master/chainerrl/links/dqn_head.py)で定義されており、
以下のように定義されています


```
class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4,
                            initial_bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias),
            L.Linear(3136, n_output_channels, initial_bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)
```

上記の通り、3層のConvolutionと一つのFCで構成されていることが分かります。

## Experience Replay

DQN の工夫は、Q関数の工夫だけではありません。それだけでは、精度がでないため、幾つかの手法が組み合わされています。ここからは、それらの手法を説明していきます。一つ目は、Experience Replayです。その名の通り、経験したものを再実行するものです。

Expirience Replayでは、経験したことを保存していきます。これは、@<m>{(S _t, a _t, r _t, S _{t + 1\\})} 、つまり現在の状態、行動、報酬、次の状態のセットで保存されています。この保存する先をReplay Bufferと言います。
このバッファからバッチサイズ分を取り出し、学習を回します。
強化学習は、単純には、ゲーム開始からゲームオーバーまでの一連のプレーをひとまとまりとして学習します。このひとまとまりをエピソードと言います。ブロック崩しのようなゲームでは、このエピソードによってプレーの傾向が異なり、学習が安定しないことがあります。Experience Replayでは、この不安定性を防ぐ為に、複数の異なるエピソードからランダムに経験を取り出し学習を行っています。

![Experience Replay](src/images/chainerrl_fig4.png)

[examples/atari/train_dqn_ale.py](https://github.com/chainer/chainerrl/blob/master/examples/atari/train_dqn_ale.py)では
以下のようにしてReplay Bufferを作っています。

```
        rbuf = replay_buffer.ReplayBuffer(10 ** 6, args.num_step_return)
```

上記のように、バッファーのサイズは100万になっている事が分かります。

## Target Network

次にTarget Netoworkについて説明します。が、これは、あまり意識することのないChainerRLの内部実装なので、簡単に触れるだけにとどめます。
この手法は、学習中のネットワークとは別に、誤差関数で利用するネットワークを用意し、誤差関数で利用するネットワークは、定期的に学習中のネットワークと同期するという手法です。
これは、DQNでは、ネットワークの学習に、そのネットワークの出力を使う事になるので、学習が安定しないということがおこるためです。

ChainerRLでは、chainerrl.agents.DQNクラスの中の、[sync_target_network()](https://github.com/chainer/chainerrl/blob/78b5e3c97f68f800057d60de361c6c220ba39bcf/chainerrl/agents/dqn.py#L221)で、その同期処理をみることができます。

```
    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)
```

## Clip reward

これはシンプルで、得られる報酬を-1, 0, 1に固定したというものです。atariのゲームでは得られるスコアが報酬となりますが、それだとゲームによって、報酬のスケールがことなり、ハイパーパラメーターチューニングをゲーム毎にやる必要が生じます。それを防ぐ為に、報酬を-1, 0, 1に固定しています。

これは、ChainerRLのサンプルのtrain_dqn_ale.pyでは、`atari_wrappers.make_atari`の`clip_reward`の設定で見ることができます。

```
    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test)
```

## その他

そろそろ手法が細かくなってきました。簡単に羅列します。

Skip frameという手法が使われていますが、単にフレームをスキップするだけです。全てのフレームを学習してもほとんど変わらないためです。

さらに手法の名前はないですが、その4フレーム内で各ピクセルでmaxを取っています。そもそも学習時はモノクロにして学習しているのですが、4フレームで一番明るい値を取るようにしているという事です。
これは、Atariはゲームはとても古いので、1画面に出せるキャラクタが限られていて、フレームによっては表示されないキャラクタがいるためです。

## train_dqn_ale.py を見てみよう

ここまで読んだら、ChainerRLのAtariのDQNのサンプルの[examples/atari/train_dqn_ale.py](https://github.com/chainer/chainerrl/blob/master/examples/atari/train_dqn_ale.py)を上から下まで読んでみてください。アルゴリズムが分かると、ソースがみるみる分かるようになったと思います。
ChainerRLはアルゴリズムの実装を肩代わりしてくれるため、とても簡単に使えますが、使いこなすには、そのアルゴリズムを知っておく必要があります。ぜひ、元論文のChainerRLのソースコードを見比べながら、各アルゴリズムの学習を楽しんでください。