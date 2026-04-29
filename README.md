# AI-Based-Cosmic-Rejection-Event-Classification-for-LArTPC-Detectors
## 1 Introduction
Neutrinos are fundamental subatomic particles that carry no electrical charge and have
very small masses. Because they interact very weakly with matter, detecting them
directly is highly difficult. As a result, the majority of the data obtained in neutrino
experiments consists of background noise originating from atmospheric cosmic rays (espe
cially muons). Particularly in detectors located near the surface, the low signal-to-noise
ratio makes it difficult to reliably distinguish rare neutrino interactions. Therefore,
effective cosmic background rejection is one of the most critical data processing steps in
modern neutrino experiments.
In traditional approaches, signal-background separation is mostly performed using cut
based and basic statistical methods. However, with increasing data volumes and multi
dimensional data structures, these methods fall short in terms of both computational cost
and classification performance.
In this study, to solve the signal-background separation problem encountered in LArTPC
based neutrino experiments, synthetic data was generated using the Geant4 simulation
toolkit . Physically meaningful features were extracted from the obtained data, and
deep learning and machine learning-based models were developed using these features.
The experimental setup and detector design were inspired by LArTPC-based neutrino ex
periments such as MicroBooNE and other large-scale neutrino observatories. The
main goal of this study is to develop a highly accurate and scalable signal-background sep
aration approach with a lower computational cost compared to traditional reconstruction
based and image-based deep learning methods.
## 1.1 Problem Statement
The rarity of neutrino interactions and the fact that detector data largely consists of
background noise from atmospheric cosmic rays make signal-background separation a
very difficult problem. Especially in large datasets, manual cuts and simple statistical
methods produce time-consuming and low-accuracy results.
While the IceCube experiment generates approximately 100 terabytes of data annually,
the vast majority of the signals consist of background noise. This situation creates
a serious bottleneck in the analysis process. Similarly, in LArTPC-based experiments,
increasing data volume and event complexity limit the scalability of current methods.
Therefore, there is a need for new methods that can operate with high accuracy and low
computational cost.
## 1.2 Contributions
In this study, synthetic data was generated using Geant4 simulations, and distinguishing
physical features such as particle arrival angle, ionization energy loss (dE/dx), spatial
distribution, and event topology were extracted from this data.
Feature selection was applied to the obtained data to determine the most effective pa
rameters for signal-background separation. Using these features, XGBoost, LightGBM,
Random Forest, CatBoost, and Deep Neural Network (DNN) models were trained and
compared.
As aresult of the analyses, it was shown that the optimized models achieved high accuracy
rates and provided advantages in terms of computational cost. This study presents an
optimized approach that addresses both classification performance and computational
efficiency together.

## 2 Methodology
In this study, an artificial intelligence-based data analysis framework was established to
detect pure neutrino signals by rejecting the cosmic ray background in neutrino experi
ments. The methodology of the research consists of five main sequential stages: synthetic
data generation, data preprocessing, feature extraction, dimensionality reduction, and the
training of machine and deep learning models

<img width="704" height="296" alt="image" src="https://github.com/user-attachments/assets/e12f0c89-a52e-44a0-894a-1cf6ddf5358f" />
<br><br>

<img width="900" height="569" alt="image" src="https://github.com/user-attachments/assets/22b8448f-4798-4e4e-90d1-add650b1f707" />
<br><br>

<img width="866" height="507" alt="image" src="https://github.com/user-attachments/assets/5e061c8b-f845-41f8-8e9e-1001a955a318" />
