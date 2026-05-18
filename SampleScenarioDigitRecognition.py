"""
SAMACT Framework
Copyright (C) 2026 Maviss Design

Licensed under the GNU Affero General Public License v3.0 (AGPLv3).

If you modify this software and make it available over a network,
you must provide the corresponding source code to users (AGPLv3 §13).

For commercial licensing options, contact:
solution-sales@maviss-design.com

See the LICENSE file for full details.
"""

# 想定実行時間
# 6-7分

# 追加ライブラリ
# scikit-learn >= 1.6.0
# コマンド例: python3 -m pip install scikit-learn

import time

import numpy as np
# データセットはsklearnのDigitsを使用。
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from samact import *

def main():
    """
    sklearnのDigitsデータセットを用いて
    3層のSAMACTが手書き数字を学習し、推論するサンプルシナリオ
    
    1. データの準備
    2. ニューラルネットワークの構築
      2-1. エンコーダの設定
      2-2. SAMレイヤーの設定
      2-3. デコーダの設定
      2-4. ニューラルネットワークの生成
    3. 学習
      3-1. 学習プロパティの設定
      3-2. 学習の実施
      3-3. 学習結果の保存
    4. 推論
      4-1. テスト用データセットをまとめて推論
      4-2. 1データを推論
    """
    # region 1. データの準備
    digits = load_digits()
    data:np.ndarray = np.array(digits.data)
    normData = data / 16.0 # 0-16のデータなので、16で割ると正規化できる。
    # normDataは(1797, 64)で、(データ数, 説明変数)の次元。

    label:np.ndarray = np.array(digits.target)

    # 学習8:テスト2に分割。再現性のためにシードは固定。
    trainData, testData, trainLabel, testLabel = train_test_split(normData, label, test_size=0.2, random_state=0)
    # endregion

    start = time.time()

    # region 2. ニューラルネットワークの構築
    # 64-32-10のネットワークを構成
    nN1 = 64
    nN2 = 32
    nN3 = 10

    ## 2-1. エンコーダの設定
    # SAMACTは主にRateCodingを採用。
    inputLayer = RateEncodeLayer(nN1)

    ## 2-2. SAMレイヤーの設定
    # ipynb側と異なり、乱数分布を指定していない。(そのため正解率が異なる)
    # ハイパーパラメータa=3, p=0.75を指定。
    # 重みの逆伝搬時の活性化関数にStep関数を指定。
    # 教師信号生成時の活性化関数にStep関数を指定。(前段はエンコーダなので利用されない)
    hiddenLayer = SAMLayer(nN2, LayerProperty(a=3, p=0.75), Step(), Step())

    # ハイパーパラメータa=3, p=0.75を指定。
    # 重みの逆伝搬時の活性化関数にStep関数を指定。
    # 教師信号生成時の活性化関数にLinear関数を指定。(特許取得済み)
    outputLayer = SAMLayer(nN3, LayerProperty(a=3, p=0.75), Step(), Linear())

    ## 2-3. デコーダの設定
    # 分類問題なので、出力パルスを多数決するデコーダを指定。
    decoder = MajorityDecodeLayer()

    ## 2-4. ニューラルネットワークの生成
    # 上記layerを渡してモデルを構築
    model = Sequential(inputLayer, decoder, [hiddenLayer, outputLayer])
    # 1データを32サイクルで処理するように設定。(ハイパーパラメータ)
    model.Compile(32)
    # endregion

    # region 3. 学習
    ## 3-1. 学習プロパティの設定
    # 学習率eta, iotaの初期値を5, 2に設定。1エポックごとに1減衰する。
    learnProperty = LearningProperty(eta=5, iota=2, decayPeriod=1)
    ## 3-2. 学習の実施
    # 学習エポック数は6に設定。
    fitResult = model.Fit(trainData, trainLabel, 10, learnProperty)

    ## 3-3. 学習結果の保存
    # パラメータの学習結果を保存
    dumpName = f"digits_{nN1}_{nN2}_{nN3}_ntd{len(trainData)}_step.h5"
    model.Save(dumpName)

    # 学習にかかった時間
    print("fit time")
    print(time.time()-start)

    # 1epoch目から20epochまでのの学習精度(正解率)のリスト
    print(fitResult.metrics)
    # endregion

    # region 4. 推論
    ## 4-1. テスト用データセットをまとめて推論
    # テストデータに対する正解率
    evalResult = model.Evaluate(testData, testLabel)
    print(evalResult.metrics)

    ## 4-2. 1データを推論
    # 1データに対して予測するときのメソッド
    predict = model.Predict(testData[0])
    print(f'Infered 1 data. predicted {predict}, label is {testLabel[0]}')
    # endregion

if __name__ == '__main__':
    main()