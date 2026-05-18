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
# 2-3分

# 追加ライブラリ
# librosa
# コマンド例: python3 -m pip install librosa

from pathlib import Path
import time

import numpy as np
# データセットは動物の鳴き声(おもちゃの音を録音)を利用。
# 前処理として、周波数解析を行うために、librosaを利用。
import librosa
from samact import *

# 前処理とmainで共通して利用する変数
N_N1 = 100 # 入力層の数

# 前処理で正規化するときに使う範囲。(全次元共通)
# 個別に情報量最大になる範囲を設定すると、より良い結果を得られる。
MIN_MEL = -50
MAX_MEL = 0

def MakePreProcessedDataset(basePath:Path)->tuple[np.ndarray, np.ndarray]:
    """
    data, labelを返す。
    この返り値をSAMACTに渡せばよい。
    """

    dataset = []
    labels = []

    # 探索順の再現性のためにsortedを使用(iでラベルを貼るため)
    # 各種wavファイルのmelspectrogramを計算し、正常/異常のラベルを貼って、行列としてまとめる。
    for i, labelDirPath in enumerate(sorted(basePath.glob("*/"))):
        datasetOnLabel = []

        # ファイルの順番の再現性のためにsortedを使用
        for wav in sorted(labelDirPath.glob("*.wav")):
            y, sr = librosa.load(path=wav, sr=None)
            melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_N1,
                                                     n_fft=2048, hop_length=256)
            melSpecDb = librosa.power_to_db(melSpec, ref=1.0,
                                            amin=1e-10, top_db=80.0)
            # librosaの返り値は想定する行列と、軸が逆なので転置してスケーリング
            normMelSpecDb = (melSpecDb.T - MIN_MEL) / (MAX_MEL - MIN_MEL)

            datasetOnLabel.append(normMelSpecDb)

        naDatasetOnLabel = np.vstack(datasetOnLabel)
        dataset.append(naDatasetOnLabel)

        labels.extend([i]*len(naDatasetOnLabel))

    # SAMACTはオンライン学習するため、ラベルの偏りが悪影響を与える。
    # シャッフルすることで対策する。
    naDataset = np.vstack(dataset)
    naLabels = np.array(labels)
    np.random.seed(0)
    indice = np.arange(len(naDataset))
    np.random.shuffle(indice)
    datasetShuffled = naDataset[indice, :]
    labelShuffled = naLabels[indice]

    return datasetShuffled, labelShuffled

def main():
    """
    動物の鳴き声(牛、熊)のデータセットを用いて
    3層のSAMACTが音声の特徴を学習し、推論するサンプルシナリオ
    牛の音声は時間が長いので、学習用データセットのファイル数を熊の半分にした。
    
    1. データの準備(前処理含む)
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
    # region 1. データの準備(前処理含む)
    trainData, trainLabel = MakePreProcessedDataset(Path("data/AnimalCry/train"))
    testData, testLabel = MakePreProcessedDataset(Path("data/AnimalCry/test"))

    # endregion
    start = time.time()

    # region 2. ニューラルネットワークの構築
    # 100-200-2のネットワークを構成
    nN1 = N_N1
    nN2 = 200
    nN3 = 2

    ## 2-1. エンコーダの設定
    # SAMACTは主にRateCodingを採用。
    inputLayer = RateEncodeLayer(nN1)

    ## 2-2. SAMレイヤーの設定
    # ハイパーパラメータa=3, p=0.75を指定。
    # 重みの逆伝搬時の活性化関数にStep関数を指定。
    # 教師信号生成時の活性化関数にStep関数を指定。(前段はエンコーダなので利用されない)
    hiddenLayer = SAMLayer(nN2, LayerProperty(a=3, p=0.75), Step(), Step())

    # ハイパーパラメータa=3, p=0.75を指定。
    # 重みの逆伝搬時の活性化関数にStep関数を指定。
    # 教師信号生成時の活性化関数にStep関数を指定。
    outputLayer = SAMLayer(nN3, LayerProperty(a=3, p=0.75), Step(), Step())

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
    # 学習エポック数は10に設定。
    fitResult = model.Fit(trainData, trainLabel, 10, learnProperty)

    ## 3-3. 学習結果の保存
    # パラメータの学習結果を保存
    dumpName = f"sound_{nN1}_{nN2}_{nN3}_ntd{len(trainData)}_step.h5"
    model.Save(dumpName)

    # 学習にかかった時間
    print("fit time")
    print(time.time()-start)

    # 1epoch目から10epochまでのの学習精度(正解率)のリスト
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