# Odecomp

In this work, for online tensor decomposition, Odecomp is proposed, which will decompose the tensors before/during the procedure of training, as efforts to improve the performance of training and the accuracy of inference. For the pre-training tensor decomposition, various approaches will be evaluate and for the in-training tensor decomposition, several strategies, incremental rank reduction and rank fading, will be applied to enable gradient descent algorithm for tensor decomposition and improve the overall performance of decomposition.

### 0. Acknowledgement

Currently, part of the source code is open to the community, through which reader might be able to get the fundamental idea of Odecomp. This project is begun as the course project of ELEC5470 Convex Optimization, HKUST. Thank Prof. Palomar, TA Junyan LIU and TA Ziping ZHAO a lot for their patience and time dedicated into this interesting course.


### 1. Highlights of Odecomp

   1) Pre-training decomposition is implemented to make full use of the process of training, to widely search for proper
solution for the decomposition. Also, this method can also reduce the training time significantly because of the reduction
of parameters.

   2) A accuracy prediction model is designed to predict the accuracy during decomposition and training. With such model,
Odecomp can determine the proper time to stop fine-tuning and shrink the consumption of time.

<p align="center">
  <img src="https://github.com/zslwyuan/Odecomp/blob/master/Result_figures/exp_predict0.png" width="400">
</p>

   3) Rank-fading is introduced to guide the training model to obtain layers which tend to be low-rank and are friendly to
tensor decomposition, by penalizing the parts in matrices which will be truncated for lower rank later. With this technique, gradient descent algorithm can be applied in the application of rank reduction. This technique can reduce the loss of accuracy after tensor decomposition and save the time required by fine-tuning.

<p align="center">
  <img src="https://github.com/zslwyuan/Odecomp/blob/master/Impl_figures/reduction.png" width="400">
</p>


   4) Incremental decomposition, inspired by incremental pruning, is presented to lower the rank of layers gradually and
fine-tune the model along with the steps of decomposition, to obtain higher accuracy.

<p align="center">
  <img src="https://github.com/zslwyuan/Odecomp/blob/master/Impl_figures/interaction.png" width="700">
</p>

### 2. Preliminary Results of Odecomp


The experiments show the significant improvement of accuracy and the reduction of traing/fine-tuning time. 

<img src="https://github.com/zslwyuan/Odecomp/blob/master/Result_figures/truncate15.png" width="400"><img src="https://github.com/zslwyuan/Odecomp/blob/master/Result_figures/truncate10.png" width="400">


<p align="center">
  <img src="https://github.com/zslwyuan/Odecomp/blob/master/Result_figures/trainingtime.png" width="600">
</p>

