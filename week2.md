+++
title = "Deep Learning Week 2"
hascode = true
date = Date(2018, 12, 30)
tags = ["deep learning", "machine learning", "notes"]
+++

# Deep Learning Course Notes - Week 2

\toc

## Introduction

I recently started the Deep Learning course by Andrew Ng. In the second week, we discuss logistic regression as a toy example to introduce basic concepts, such as forward and reverse propagation, computational graph, and gradient descent.

I usually remember better if I try explaining concepts to someone else, hence these notes serve such a purpose and follow my understanding of the course content.

## Relation to Linear Regression

In logistic regression, the dependent variable is modelled as a probability of seeing a binary outcome.
We start with linear regression model which assumes continuous normally distributed values of dependent variable.

\begin{equation}
\mathbf{y} = \mathbf{X\beta} + \mathbf{\epsilon}
\end{equation}

Similarly, for logistic regression we call coefficients as weights \(\mathbf{w}\), drop error term from the model and apply the sigmoid function \(\Sigma\):

\begin{equation}
\Sigma(z)=\frac{1}{1+e^{-z}}
\end{equation}

\begin{equation}
z = \mathbf{W x}+\mathbf{b}
\end{equation}

where \(\mathbf{b}\) is the intercept term. In this course, intercept is modelled separately and not as the usual notation with column of ones.

Important to note is that we can now use the familiar linear regression model with sigmoid function applied after computing dot product to get a logistic regression. This is particularly important later when we compute gradients and update model parameters.

## Loss and Cost Function

We defined *loss function* as logistic function for a single example:

\begin{equation}
\mathcal{L}(\hat{y}^{(i)},y^{(i)}) = - (y^{(i)} log\hat{y}^{(i)} + (1-y^{(i)})log(1-\hat{y}^{(i)}))
\end{equation}

To calculate overall cost across all training examples, we define *cost function* as:

\begin{equation}
J(w,b) = \frac{1}{m}\sum\limits^m_{i=1}\mathcal{L}(\hat{y}^{(i)},y^{(i)})
\end{equation}

which is just the average of outcomes from multiple loss functions:

\begin{equation}
J(w,b) = -\frac{1}{m}\sum\limits^m_{i=1}(y^{(i)} log\hat{y}^{(i)} + (1-y^{(i)})log(1-\hat{y}^{(i)}))
\end{equation}

## Gradient Explained

A little bit of digression here. I was wondering how gradient descent relates to normal equations.
I found that normal equations require calculating inverse which must exist and calculating that inverse is higher computational complexity than calculating gradients (for proof, google yourself).

For non-convex problems, stochastic gradient descent can be used (not encountered yet in this course).
For convex problems such as logistic regression, we use gradient descent. Parameters \(\mathbf{w}\) are repeatedly updated, given \(\alpha\) learning rate. 

Learning rate determines how large steps we take to arrive at minimum. How to tune this parameter is nicely explained later in the course.

Derivative (slope) of a function can be positive or negative, determining if \(w\) should be increased or decreased. For more parameters it is a partial derivative of course.

\begin{equation}
w = w - \alpha \frac{\delta J(w,b)}{\delta w}
\end{equation}

\begin{equation}
b = b - \alpha \frac{\delta J(w,b)}{\delta b}
\end{equation}

Recall that parameters must be updated simultaneously.

## Computational Graph

Computational graph is a way to calculate derivatives for more complicated functions.

- Forward step keeps information how calculations were performed and associated values
- Backward step calculates derivatives

Simple toy example:
\begin{equation}
J(a,b,c) = 3(a+bc)
\end{equation}

Forward steps, keeps single operations and associated values computed from initial \(a,b,c\) parameters:
\begin{equation}
u = bc
\end{equation}
\begin{equation}
v = a+u
\end{equation}
\begin{equation}
J = 3v
\end{equation}

Backward steps, starts from the right, computes derivative for last the operation:
\begin{equation}
\frac{dJ}{dv} = 3
\end{equation}

then we keep going backwards and use chain rule to calculate rate how \(J\) changes when we change \(a\) for each step:
\begin{equation}
\frac{dJ}{da} = \frac{dJ}{dv}\frac{dv}{da}
\end{equation}

which is just a multiplication of computed derivatives from the right. This way we arrive at the slopes for the input variables which are our *gradients*.

Recall that gradient tells us if we should increase or decrease parameter we aim to update and how "fast" by steepness of the slope.

## Vectorization

In order to take advantage of hardware vector operations, we must avoid using loops. 

My colleague had an argument that in compiled programming languages like Julia loops are efficient and it's not necessary to vectorize the code. My *opinion* is that it probably depends on how good the compiler is in guessing how to vectorize your code and it might pay off to do it right from the start.

In logistic regression we will need to compute for \(n\) features and \(m\) examples many:

\begin{equation}
z^{(n)} = w^tx^{(n)}+b
\end{equation}
\begin{equation}
a^{(n)} = \sigma(z^{(n)})
\end{equation}

so for each \(n\)-th feature we find a \(w,b\) coefficients.

This can be simplified using matrix notation by stacking \(x^{(n)}\) vectors as columns as:
\begin{equation}
[z \ldots ] = [w \ldots]^t \mathbf{X} + [b \ldots]
\end{equation}

equivalent to:
\begin{equation}
\mathbf{z} = \mathbf{w}^t \mathbf{X} + \mathbf{b}
\end{equation}

Note, \(b\) is 1 by 1 matrix, but thanks to **broadcasting** in python it gets expanded to 1 by m vector, matching the dimensions of the earlier dot product.

## Vectorized Computation for Gradients

Recall that for one training example we had cost function as an average of individual losses:
\begin{equation}
J(w,b) = \frac{1}{m} \sum\limits^m_{i=1} \mathcal{L}(a^{(i)}, y^{(i)})
\end{equation}

where \(a\) is:
\begin{equation}
a^{(i)} = \hat{y}^{(i)} = \sigma(z^{(i)}) = \sigma(w^t x^{(i)}) + b)
\end{equation}

with \(i\) being examples and \(\sigma\) a sigmoid function.

To obtain gradients given our overall cost function including \(w\) parameters we want to estimate, we take derivative of the whole cost function with respect to parameters \(w\):
\begin{equation}
\frac{\delta J}{\delta w} J(w,b)
\end{equation}

This is equivalent to average of computed derivatives for individual losses, which we know already how to calculate with a help of computational graph:
\begin{equation}
\frac{\delta J}{\delta w} J(w,b) = \frac{1}{m} \sum\limits^m_{i=1} \underbrace{\frac{\delta J}{\delta w} \mathcal{L}(\sigma(w^t x^{(i)}) + b), y^{(i)})}_{dw^{(i)}}
\end{equation}

Simplifying, to compute derivative for \(w\) parameter we just compute:
\begin{equation}
dw = \frac{1}{m} \sum\limits^{m}_{i=1} dz^{(i)}
\end{equation}

and from the course using computational graph we found that:
\begin{equation}
dz^{(i)} = a^{(i)} - y^{(i)}
\end{equation}

Therefore to find derivative for J with respect to \(w\) we can calculate:
\begin{equation}
dw = \frac{1}{m} \sum\limits^{m}_{i=1} x^{(i)}(a^{(i)} - y^{(i)})
\end{equation}

Note \(x\) in the above equations, in the course we do \(J(a,b,c)\) example with just parameter we estimate, so I got lost a bit why we multiply by \(x\) now which is our data (we don't estimate).
I am not sure if this is perfectly correct, but my intuition is that derivative must depend on \(x\) at any point and therefore we multiply to obtain \(dw\).

To vectorize these operations we can compute derivatives with respect to \(b\) and \(w\):
\begin{equation}
db = \frac{1}{m} \sum\limits^m_{i=1} dz^{(i)}
\end{equation}

and:
\begin{equation}
dw = \frac{1}{m} \mathbf{X} dZ^t
\end{equation}

and use these to update parameters in single iteration of gradient descent:
\begin{equation}
w := w - \alpha dw
\end{equation}
\begin{equation}
b := b - \alpha db
\end{equation}

