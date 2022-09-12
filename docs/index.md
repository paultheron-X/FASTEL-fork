# Flexible Modeling and Multitask Learning using Differentiable Tree Ensembles
** Authors: Shibal Ibrahim, Hussein Hazimeh, Rahul Mazumder **

This site provides an introduction to FASTEL, a new toolkit for learning differentiable tree ensembles (2022, Ibrahim, Hazimeh and Mazumder). An introduction to differentiable tree ensembles can be found below. To dive in, a code tutorial can be found in the [Tutorial Section](tutorial.md). 

Our contributions, which can be summarized as follows, are:

   - Proposition of a flexible framework for training differentiable tree ensembles with seamless support for new loss functions.
   - Introduction of a novel, tensor-based formulation for differentiable tree ensembles that allows for efficient training on GPUs.
   - Extension of differentiable tree ensembles to multi-task learning settings by introducing a new regularizer that allows for soft parameter sharing across tasks.
   - Introduction of FASTEL — a new toolkit (based on Tensorflow 2.0) for learning differentiable tree ensembles

To have more details about our countributions, please visit [Fastel](fastel.md). 

## Introduction to differentiable tree ensembles

We learn an ensemble of m differentiable trees. Let  $f^j$ be the $j$ th tree in the ensemble. For easier exposition, we consider a single-task regression or classification setting—see Section 5 for an extension to the multi-task setting. In a regression setting $k=1$, while in multi-class classification setting $k = C$, where $C$ is the number of classes. For an input feature-vector $x \in \mathbb{R}^p$ , we learn an additive model with the output being sum over outputs of all the trees:

\begin{equation}
f(x) = \sum_{j=1}^m f^j(x)
\end{equation}

The output, $f(x)$, is a vector in $\mathbb{R}^k$ containing raw predictions. For multiclass classification, mapping from raw predictions to $Y$ is done by applying a softmax function on the vector $f (x )$ and returning the class with the highest probability. Next, we introduce the key building block of the approach: differentiable decision tree.

##### Differentiable decision trees
Classical decision trees perform hard sample routing, i.e., a sample is routed to exactly one child at every splitting node. Hard sample routing introduces discontinuities in the loss function, making trees unamenable to continuous optimization. Therefore, trees are usually built in a greedy fashion. In this section, we first introduce a single soft tree and extended to soft tree ensembles. A soft tree is a variant of a decision tree that performs soft routing, where every internal node can route the sample to the left and right simultaneously, with different proportions. This routing mechanism makes soft trees differentiable, so learning can be done using gradient-based methods.
Let us fix some $j \in [m]$ and consider a single tree $f^j$ in the additive model. Recall that $f^j$ takes an input sample and returns an output vector (logit), i.e., $f^j : X ∈ \mathbb{R}^p → \mathbb{R}^k$ . Moreover, we assume that $f^j$ is a perfect binary tree with depth $d$. We use the sets $\mathcal{I}^j$ and $\mathcal{J}^j$ to denote the internal (split) nodes and the leaves of the tree, respectively. For any node  $i \in \mathcal{I}^j \cup \mathcal{J}^j$ , we define $A^j(i)$ as its set of ancestors and use the notation $x → i$ for the event that a sample $x \in \mathbb{R}^p$ reaches $i$.

##### Routing
Internal (split) nodes in a differentiable tree perform soft routing, where a sample is routed left and right with different proportions. This soft routing can be viewed as a probabilistic model. Although the sample routing is formulated with a probabilistic model, the final prediction of the tree $f$ is a deterministic function as it assumes an expectation over the leaf predictions. Classical decision trees are modeled with either axis-aligned splits
or hyperplane (a.k.a. oblique) splits. Soft trees are based on hyperplane splits, where the routing decisions rely on a linear combination of the features. Particularly, each internal node $i \in \mathcal{I}^j$ is associated with a trainable weight vector $w_i^j \in \mathbb{R}^p$ that defines the node’s hyperplane split. Given a sample $x \in \mathbb{R}^p$ , the probability that internal node i routes $x$ to the left is defined by $S(w_i^j ·x)$,where $S : \mathbb{R} → [0, 1]$ is an activation function. Now we discuss how to model the probability that $x$ reaches a certain leaf $l$. Let $[l \swarrow i]$ (resp. $[i \searrow l]$) denote the event that leaf $l$ belongs to the left (resp. right) subtree of node $i \in \mathcal{I}^j$. Assuming that the routing decision made at each internal node in the tree is independent of the other nodes, the probability that x reaches l is given by:

\begin{equation}
P^j(\{x → l\}) = \prod_{i \in A^j(l)} r_{i,l}^j(x)
\end{equation}

where $r_{i,l}^j(x) = S(w_i^j ·x)1[l \swarrow i]\bigodot(1 − S(w_i^j ·x))1[i \searrow l]$. The probability of node $i$ routing $x$ toward the subtree containing leaf l.
Popular choices for $S$ include logistic function and smooth-step function (for conditional computation as in classical trees with oblique splits). 

#### Prediction
As with classical decision trees, we assume that each leaf stores a weight vector $o_l^j \in \mathbb{R}^k$ (learned during training). Note $j$ that, during the forward pass, $o_l^j$ is a constant vector, meaning that it is not a function of the input sample(s). For a sample $x \in \mathbb{R}^p$ , we define the prediction of the tree as the expected value of the leaf outputs, i.e.,

\begin{equation}
f^j(x) = \sum_{l \in L} P^j(\{x → l\}) o_l^j
\end{equation}

where $L$ is the set of leaves in the tree

#### Conclusion
End-to-end learning with differentiable tree ensembles appears to have several advantages. 

 1. Training is easy to set up in public deep learning frameworks. Differentiable tree ensembles allow for flexibility in loss functions without the need for specialized algorithms. For example, mixture likelihoods can be easily implemented in Tensorflow Probability, which allows for handling zero-inflated data. Similarly, multi-task loss objectives can also be handled. 
 2. With a careful implementation, the tree ensemble can be trained efficiently on GPUs — this is not possible with earlier toolkits such as TEL.
 3. Differentiable trees can lead to more expressive and compact ensembles. This can have important implications for interpretability, latency and storage requirements during inference.

This is the framework implemented in the FASTEL package. To see it in action, the tutorial can be found [here](tutorial.md). 