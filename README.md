# GDMAM: Gradient Decomposition Multi-Momentum Allocation Method for Imbalanced Label Distribution Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper:

**GDMAM: Gradient Decomposition Multi-Momentum Allocation Method for Imbalanced Label Distribution Learning**

> **Abstract:** *Label Distribution Learning (LDL) is an effective paradigm for addressing tasks with label ambiguity. However, the subjectivity in annotating label description degrees often results in imbalanced label distributions. Existing methods typically combine distribution-alignment loss with label-prediction loss to alleviate the feature shift introduced by Imbalanced Label Distribution Learning (ILDL), but optimizing these objectives with a unified strategy often leads to gradient conflicts and slows the convergence of the prediction branch. To address this issue, we propose the Gradient Decomposition Multi-Momentum Allocation Method (GDMAM). GDMAM decomposes the alignment gradients into parallel and orthogonal components and assigns each component an adaptive momentum buffer that is updated based on its training dynamics. In addition, we introduce a gradient-norm-based dynamic weighting mechanism to better balance the optimization objectives and improve training stability. Extensive experiments under various imbalance settings demonstrate that GDMAM significantly accelerates convergence and outperforms strong ILDL baselines.*

## 🚀 Framework

![GDMAM Framework](assets/framework.png)
*Fig 1. Overview of GDMAM. (a) Gradient Decomposition: The alignment gradient is decomposed into parallel and orthogonal components along the prediction gradient. (b) Component Momentum Allocation: Independent momentum buffers are assigned with adaptive weights. (c) Momentum update rule.*

## ✨ Key Features

* **Gradient Decomposition:** explicitly separates cooperative ($g_{\parallel}$) and conflicting ($g_{\perp}$) gradient components between alignment and prediction tasks.
* **Multi-Momentum Allocation:** Assigns independent momentum buffers (EMA) to decoupled gradients to stabilize optimization.
* **Adaptive Weighting:** Uses a gating mechanism based on gradient statistics ($\sigma$) to dynamically balance contributions.
* **Plug-and-Play:** Can be easily integrated into existing ILDL frameworks as a robust optimizer.

## 🛠️ Requirements

The code is implemented using PyTorch. We recommend using **Python 3.8+**.

1. Clone this repository:
   ```bash
   git clone [https://github.com/](https://github.com/)[YourUsername]/GDMAM.git
   cd GDMAM
