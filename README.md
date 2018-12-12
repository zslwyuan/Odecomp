# Odecomp

In this work, for online tensor decomposition, Odecomp is proposed, which will decompose the tensors before/during the procedure of training, as efforts to improve the performance of training and the accuracy of inference. For the pre-training tensor decomposition, various approaches will be evaluate and for the in-training tensor decomposition, several strategies, incremental rank reduction and rank fading, will be applied to enable gradient descent algorithm for tensor decomposition and improve the overall performance of decomposition.

### Highlights of Odecomp

   1) Pre-training decomposition is implemented to make full use of the process of training, to widely search for proper
solution for the decomposition. Also, this method can also reduce the training time significantly because of the reduction
of parameters.
   2) A accuracy prediction model is designed to predict the accuracy during decomposition and training. With such model,
Odecomp can determine the proper time to stop fine-tuning and shrink the consumption of time.
    3) Rank-fading is introduced to guide the training model to obtain layers which tend to be low-rank and are friendly to
tensor decomposition, by penalizing the parts in matrices which will be truncated for lower rank later. With this technique,
gradient descent algorithm can be applied in the application of rank reduction. This technique can reduce the loss of
accuracy after tensor decomposition and save the time required by fine-tuning.
    4) Incremental decomposition, inspired by incremental pruning, is presented to lower the rank of layers gradually and
fine-tune the model along with the steps of decomposition, to obtain higher accuracy.

(Currently, part of the source code is open to the community, through which reader might be able to get the fundamental idea of Odecomp. This project begins as the course project of ELEC5470 Convex Optimization, HKUST. Thank Prof. Palomar, TA Junyan LIU and TA Ziping ZHAO a lot for their patience and time dedicated into this interesting course.)
