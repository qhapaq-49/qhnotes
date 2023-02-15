# 課金データセットの使い方マニュアル
- 本リポジトリはQhapaq開発チームが[note](https://note.com/qhapaq_shogi)上で販売している各種データの使い方をまとめたものです。**とくに開発者向けのデータ（e.g. 追加学習が可能なpthファイル）を購入される方については事前にこちらのページを確認していただくことをお勧めします**。
  - **本プロジェクトへのお布施として購入される場合についてはその限りではありません**。

# dlshogi、ふかうら王向けの評価関数(.onnxファイル)
dlshogiや、ふかうら王から読み込むことで盤面解析・定跡作成・対局などの用途にお使いいただけます。本マニュアルではdlshogi・ふかうら王の導入方法は説明しません。例えば以下のリンクなどが役に立つことでしょう。

- [dlshogi導入の解説記事](https://kakuyasu-sim-now.com/shogi/dlshogi-install/)
- [dlshogi導入のYouTubeの動画解説](https://www.youtube.com/watch?v=ZXo_eOjyaMY)
- [ふかうら王の導入記事](https://github.com/yaneurao/YaneuraOu/wiki/%E3%81%B5%E3%81%8B%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E6%89%8B%E9%A0%86)


## テスト用評価関数
テスト用の評価関数として第三回世界電竜戦で6位になったJust Stop 26歩の評価関数を公開しています。[このリンク](https://github.com/qhapaq-49/qhapaq-bin/releases/tag/tagtest)からファイルをダウンロードして`eval/model.onnx`をdlshogi、ふかうら王で読み込むことができれば、本プロジェクトで扱っている各種onnxファイルを読み込むことも可能だと思われます。盤面解析・定跡作成・対局などをお楽しみいただければ幸いです

# 追加学習可能なdlshogi、ふかうら王向けの評価関数(.pthファイル)

本プロジェクトでは追加学習可能なpthファイルも販売しています。pthファイルを使うことでモデルのアーキテクチャや各変数などを読み取ることが容易になり、追加学習や転移学習なども行えるようになります。ただし、pthファイルを使いこなすにはdlshogiの学習部を動かすための環境が必要になります。

## 環境整備：テストしたいだけの人向け
dlshogi, cshogi, pytorchが揃っていればOKです。cudaなどのgpu環境は不要です（ただし、本当に学習を回すつもりならgpu環境は必須だと思います）

    # dlshogiのインストール
    git clone git@github.com:TadaoYamaoka/DeepLearningShogi.git
    cd DeepLearningShogi
    pip install -e .
    # cshogiのインストール
    git clone git@github.com:TadaoYamaoka/cshogi.git
    cd cshogi
    pip install -e .
    # pytorchのインストール
    pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu


## 環境整備：GPU+tensorrtなどのガチ環境を用意する人向け
[dlshogiのDockerfile](https://github.com/TadaoYamaoka/DeepLearningShogi/tree/master/docker)から環境構築をすることをお勧めします。

## テスト用評価関数
テスト用の評価関数を[こちらのリンクからダウンロードできます](https://github.com/qhapaq-49/qhnotes/releases/tag/testdata)。関数付属の`test.py`が無事に動けば導入成功です。あとはdlshogiのtrainerに突っ込むなどして追加学習を楽しんでいただければ幸いです。dummy関数向けの`test.py`(評価関数を読み込みベンチマークを行うスクリプト)と`policy_value_network_qhapaq.py`(配布した評価関数のアーキテクチャが記載されたスクリプト。配布する評価関数ごとに変わることに注意！)のコードは[dummyフォルダ](dummy)からも参照いただけます。**Qhapaqチームが配布する評価関数をdlshogiで追加学習していただくにあたり、このコードを読んで何をしているかが理解できるぐらいのdlshogiの知識があることを前提としていることをご注意ください**。dlshogiの動作原理や運用方法について興味がある方はdlshogi開発チームが販売している書籍 「[強い将棋ソフトの創りかた](https://www.amazon.co.jp/%E5%BC%B7%E3%81%84%E5%B0%86%E6%A3%8B%E3%82%BD%E3%83%95%E3%83%88%E3%81%AE%E5%89%B5%E3%82%8A%E3%81%8B%E3%81%9F-%E5%B1%B1%E5%B2%A1%E5%BF%A0%E5%A4%AB/dp/4839977348)」を購入されることをお勧めします

    # ダウンロードした関数があるフォルダに移動して
    cd dummy
    # test.pyを実行する
    python test.py