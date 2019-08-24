# Chainerにコントリビュート！

@<icon>{cheiko} 「CIFARの画像データを使って、Chainerのサンプルコードを書き換えて...!  
できた！ChainerMN用のCIFARデータセットを使った学習プログラム！」   

ちぇい子は作った学習プログラムの動作確認をした。

@<icon>{cheiko} 「よしよし、動作した。このプログラム、Chainerを使っている別の人にも気軽に使ってもらえたらいいのになぁ。」

@<icon>{cheita} 「出来るよ！そんなことも知らないの？」

@<icon>{cheiko} 「知らなかった。。(ちぇい太くんは本当に色々と知っているな、最近の小学生はすごい。)」

@<icon>{cheita} 「ChainerはOSSだから、Githubから誰でもプルリクが出来るんだ！」

@<icon>{cheiko} 「プルリク...？」

@<icon>{sensei} 「そうそう、ちぇい太君の言う通り、追加してほしい機能やプログラム、バグ修正などを"Pull request"や"Issue"にしてChainer開発者に要望をだすことが出来るんだ。ワン！」

@<icon>{cheiko} 「そんなことが出来るんだ～。でも、プルリクってハードルが高そう。。」

@<icon>{cheita} 「使っているChainerに貢献できるんだよ？それに合宿の成果をみせてよ。はやく、プルリクするよ！」

ちぇい太君に「合宿の成果」と言われて、このままでは終われない。と、ちぇい子はプルリクをするためにChainerをGithubからForkした。
  
--

この章は、MNIST[^11]以外で公開されているデータセットと学習プログラムで  [^11]: 手書き数字の画像データセット  
ChainerMN[^12]を使って学習したい。という気持ちから作成した学習プログラムをChainerにプルリクした体験談です。[^12]: Chainerの学習を分散処理する高速化機能。https://github.com/chainer/chainermn
 

## Pull Requestを作成
プルリクした学習プログラムはChainerMNでCIFAR[^13]を学習するものでした。[^13]: 80 million tiny imagesのラベル付きサブセット画像データセット。CIFAR10、CIFAR100がある。  

このプログラムを作成したキッカケについて少し触れさせてください。  
ChainerMNを使いたいと思った時に `chainer/examples/chainermn` を探したところ  
当時、画像分類系はMNISTとImageNetのサンプルコードがありました。  
お手軽にお試しできるのはMNISTだけど、"分散"深層学習には味気ないし、ImageNetは画像データを用意しないといけないからお手軽じゃない。  
...CIFARのサンプルコードほしいな。というキッカケで`chainer/examples/cifar` のコードを元に作成しました。  
そして、作成したプログラムは手元に置いておき  
Chainer側から作成されないかな～と待っていました。が！3ヵ月ぐらい待ってもリリースがなかったので、自分の作成したプログラムをマージしてもらおうと、おそるおそるプルリクをしました(笑)  

実際にプルリクした内容はこちら。

<div align="center"><img src="images/contribute_01.JPG"></div>
<div align="center">Pull requestの内容</div> 


## Reviwerとやり取り

@<icon>{cheiko} 「プルリクしたけど、その後どうすればいいの？」

@<icon>{cheita} 「本当にプルリクしたんだ。とりあえず、プルリクをレビューしてくれるReviwerを待つことだね。」

@<icon>{sensei} 「Reviwerを待つ間に、コミットしたものに"×"ってついているからエラー内容を確認してわかる範囲で修正しておくといいよ。」

プルリクをするとChainer開発者側からカテゴリーラベルがつけられました。そして、Reviwerがアサインしてくれました。  
今回、レビューを担当してくださった方は、Fukudaさん[^14]でした。[^14]: FukudaさんのGithub https://github.com/keisukefukuda


レビュワーと、レビューして頂いた内容に対してコメントやプログラムの修正を何度もやり取りをしました。  
やり取りをしていく中で、Twitterで日本語で対応するよ！と言っていただきました。 お言葉に甘えて、TwitterのDMで日本語でやり取りをしました。  
分かっていなかったことは丁寧に教えてくださいました。

<div align="center"><img src="images/contribute_02.JPG"></div>
<div align="center">レビュワーのFukudaさんとのDMやり取り</div>  
  
<br>
Githubでは、全て英語でやり取りをしなければならないですよね。プルリクのハードルが高い理由の一つでした。私の場合、英語は翻訳先生にほとんど頼んでいたため、日本語でやり取りできたのはすごく嬉しかったです。  

全てのレビュワーが日本語で対応してくれるかどうかはわかりませんが、レビュワーが日本人というだけで、少し心に余裕ができ、プルリクしてみようかなという気持ちになるのではないでしょうか。  

@<icon>{cheita} 「Chainerは日本発のフレームワークだから、日本人の開発者がいるよね。これは日本人にとって、とっても嬉しいね。」

@<icon>{cheiko} 「確かに♪ 心強い！Chainerにプルリクするハードルって他のフレームワークに比べて高くないのかも？」

## Contributerデビュー

エラーがなくなり、コミットが"〇"になったらレビュワーがchainer-ciにtestを要求し、successedが返ってきたら
コントリビュートまで秒読みです。

<br>

<div align="center"><img src="images/contribute_03.JPG"></div>
<div align="center">コミットがtest成功</div>


<br>
レビュワーに最終チェックをしてもらい、Chainerのmaster BranchにMargeしてもらうとコントリビュート完了です。  

<br>
<br>
<div align="center"><img src="images/contribute_04.JPG"></div>
<div align="center">Pull requestがMargeされる</div> 

<br>

割り当てられたマイルストーンがリリースすると  
リリースノートにプルリクの内容が書かれ、名前も載りました。
名前まで載ると嬉しかったです。自分の財産になりました。

<br>
<div align="center"><img src="images/contribute_05.JPG"></div>
<div align="center">リリースノートに載る</div> 
<br>

プルリクを提出してから、実際にMargeされるまで約1ヵ月半かかりました。
※レビュワーとやり取りを開始してからは半月ぐらいでした。

 ChainerMNで画像分類を試してみたいときに、よかったら使ってみてくださると嬉しいです。 
`chainer/examples/chainermn/cifar` にあります！

@<icon>{cheita} 「一度でもコントリビュートすると、コントリビューターデビューだよ。」

@<icon>{cheiko} 「私が、Chainerのコントリビューターに。。！合宿の成果がこんな形になるなんて思わなかった。」

 @<icon>{cheita} 「やればできるじゃん。」

@<icon>{cheiko} 「ちぇい太くん、私のことみなおした？」

@<icon>{cheita} 「まだまだだね。あとコントリビュート100回ぐらいしたらみなしてやるよ。」

ちぇい子がちぇい太くんに認められるにはまだ先のようだ。

## コントリビュートのススメ
使っているフレームワークに自分のプルリクがMargeされると嬉しいですよね。  

新しい機能や拡張、バク修正だけでなく  
簡単にChainerを試せるサンプルプログラムのバリエーションを増やすコントリビュートはいかがでしょうか。    
例えば、こんなサンプルプログラムがあったら嬉しいなーと個人的に思っています。  
- ChainerでCIFARのData parallel のプログラム

<br>
また、プログラム動かない～、分からない～、どうすればいいの～というときは、ChainerのSlack[^15]に質問してみてはいかがでしょうか。[^15]: Slackへの参加はこちらhttps://bit.ly/chainer-jp-slack   

Chaienr関係者や詳しい方々が答えてくれます！(日本語で対応してくれます、しかもレスポンス早め(有難い))    
Chainerチュートリアルの質問チャンネルもあります。  


コントリビュートする形は様々です。  
Chainerにコントリビュートして、利用者からコントリビューターになって一緒に名前を刻みませんか！  
この章を読んで、コントリビュートしてみようかなと思ってくれたら嬉しいです。

@<icon>{sensei} 「Chainerにコントリビュート待ってるワン！」
