# PySAMACT: Python API for PoC of the SAMACT Hardware Core IP

[![PyPI Version][(https://img.shields.io/pypi/v/samact.svg)](https://pypi.org/project/samact/)  
[![License](https://img.shields.io/pypi/l/samact.svg)](https://github.com/maviss-design/SAMACT/blob/main/LICENSE)

SAMACTは、メイビスデザイン株式会社が開発・保有する、オンチップ学習に対応したエンドポイント向けAI IPです。  
本ライブラリは、**SAMACTのハードウェア実装と同等の数値仕様および演算仕様を再現したPythonライブラリ**であり、主に**PoC（概念実証）用途**での利用を想定しています。  
SAMACTの詳細については、[当社Webサイト](https://maviss-design.com/download/?utm_source=samact_framework&utm_medium=github&utm_campaign=download)をご参照ください。

***

## 概要 (Overview)

本ライブラリは、以下を目的として開発されています。

*   ハードウェア（RTL / 専用回路）実装向けニューラルネットワークの  
    **事前評価を Python 上で行う**
*   **量子化・固定小数点演算**など、HW特有の挙動を再現

> ⚠️ 本ライブラリは **高速な学習・高性能推論用途を目的としていません**。

***

## 特徴 (Features)

*   ハードウェアと対応した演算仕様
    *   固定小数点 / ビット幅制約
    *   Saturation / Clipping
*   Python / NumPy ベースの実装
*   PoCでの利用を想定したシンプルなAPI

***

## 非対象 (Non-Goals)

以下は明示的にスコープ外です。

*   GPU / 高速化最適化
*   PyTorch / TensorFlow 互換 API の完全再現
*   自動微分

***

## インストール (Installation)

```bash
pip install samact
```

***

## 想定ユースケース (Use Cases)

*   機械学習プロジェクトのPoC
*   RTL 実装前の数値シミュレーション

***

## API仕様書 (API Document)
API仕様書は下記をご覧ください。  
XXXXXX(github pagesへのリンク)  
※準備中

## 制限事項 (Limitations)

*   実行速度は最適化されていません
*   ソースコードはPyPIで公開されており、githubでは公開されていません。

***

## ライセンス (License)

AGPLv3です。    
詳細は [LICENSE](https://github.com/maviss-design/SAMACT/blob/main/LICENSE) を参照してください。  

***

## 注意事項

本ライブラリは **研究・検証用途**を目的としています。  
商用品質・量産用途への直接利用は想定していません。

***


# 英語

# PySAMACT: Python API for PoC of the SAMACT Hardware Core IP

https://img.shields.io/pypi/v/samact.svg](https://pypi.org/project/samact/)  
https://img.shields.io/pypi/l/samact.svg](LICENSE (GitHub path))

SAMACT is an AI IP for endpoint devices with on-chip learning capability, developed and owned by Maviss Design Inc.  
This library is a **Python implementation that reproduces the numerical specifications and computational behavior equivalent to the SAMACT hardware implementation**, and is primarily intended for **Proof of Concept (PoC)** use.

For more details on SAMACT, please refer to our [website](https://maviss-design.com/download/?utm_source=samact_framework&utm_medium=github&utm_campaign=download)

***

## Overview

This library is developed for the following purposes:

*   To **perform pre-evaluation of neural networks on Python** before hardware (RTL / dedicated circuit) implementation  
*   To reproduce hardware-specific behavior such as **quantization and fixed-point arithmetic**

> ⚠️ This library is **not intended for high-speed training or high-performance inference**.

***

## Features

*   Computation specifications aligned with hardware
    *   Fixed-point arithmetic / bit-width constraints
    *   Saturation / Clipping
*   Python / NumPy-based implementation
*   Simple API designed for PoC use

***

## Non-Goals

The following are explicitly out of scope:

*   GPU acceleration / performance optimization
*   Full compatibility with PyTorch / TensorFlow APIs
*   Automatic differentiation

***

## Installation

```bash
pip install samact
````

***

## Use Cases

*   PoC for machine learning projects
*   Numerical simulation prior to RTL implementation

***

## API Documentation

Please refer to the following for the API documentation:  
XXXXXX (link to GitHub Pages)  
Now deploying

***

## Limitations

*   Execution speed is not optimized
*   Source code is distributed via PyPI and is not publicly available on GitHub

***

## License
AGPLv3.  
See [LICENSE](https://github.com/maviss-design/SAMACT/blob/main/LICENSE). 

***

## Notes

This library is intended for **research and validation purposes**.  
It is not designed for direct use in production-quality or mass-production applications.

***
