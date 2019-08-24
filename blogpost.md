# Interpretable Machine Learning / Explainable Artificial Intelligence

Automatization is a good thing in general. It finishes easy, repetitive tasks, and enables humans to use their time and brain power for newer, more challenging problems. And since machine learning (ML) models automatize decisions, they are a good thing in general, as well.

However, there are a few problems with these models, especially the more complex ones (also called *black-box models*).

## Black-Box Models have Problems

Three stories shall illustrate the common problems with black-box models.

### 1. Fairness

In October 2018 world headlines reported about [Amazon AI recruiting tool](https://www.theguardian.com/technology/2018/oct/10/amazon-hiring-ai-gender-bias-recruiting-engine) that favored men. Amazon’s model was trained on biased data that were skewed towards male candidates. It has built rules that penalized resumes that included the word “women’s”. ([Source blog post](https://appsilon.com/please-explain-black-box/))

### 2. Understanding

More and more scientific disciplines (like biology) use ML models for producing scientific outcomes. In these cases, we want to use data to extract *scientific knowledge*. We don't care too much about a 98% accurate Random Forest if we can't extract an explanation from the model ([Roscher 2019](https://arxiv.org/pdf/1905.08883.pdf))

### 3. Explainability and Debugging

The EU's GDPR states: "[the data subject should have] the right ... to obtain an explanation of the decision reached". 

One of the consequences of this law is that in February 2019, the Polish government added an amendment to a banking law that says that a bank needs to be able to explain why a loan wasn’t granted if the decision process was automatic.

A similar problem is that an inaccurate model must be *debugged*, e.g. to find out why some data was misclassified. If you know why and where your model fails, you're better equipped for feature engineering, or can better decide to replace a model with a different one.

One good example is from [Ribeiro et al. 2016](https://arxiv.org/pdf/1602.04938.pdf), a classification problem of "Husky or Wolf?", where some Huskys were misclassified. If you look at the pixels that influenced the decision the most, you find that the model has learned to use the snow in the background as a strong indicator for "Wolf":

<img src="img/husky-vs-wolf-LIME-paper.png" style="height: 200px" />

In this case, you'd have learned that one possible way forward would be to collect more training examples from Huskys in winter.

## Interpretable Machine Learning (IML) to the rescue!

> "Interpretability is the degree to which a human can understand the cause of a decision"
>
> -- Miller, Tim. "Explanation in artificial intelligence: Insights from the social sciences." [arXiv:1706.07269](https://arxiv.org/abs/1706.07269)

The lesson in these three stories is that the recent rise of powerful but complex black-box models introduces the need to be able to look under the hood of machine learning models and *understand* their behavior.

Optimizing a loss function is easy to operationalize, and that's one reason why we frame a ML problem in terms of a loss that we want to minimize. But in reality, we often care about more criteria than just accuracy ([Doshi-Velez et al., 2017](http://arxiv.org/abs/1702.08608)): We want our model to be **fair** and **explainable** - this would also increase social acceptance of our model, and of AI in general. We also care a lot about **safety**. In autonomous driving, we want to be 100% certain that the abstraction for "cyclist" a neural network has learned is correct. Imagine a model would need to "see" two wheels to recognize a bicycle; then, a bicycle with bags over its back wheel would get run over by a car.

The field of Interpretable Machine Learning (IML) is set out to solve these problems. It's a relatively new field, and as such, it changes quickly, new methods emerge at high speed, and the terminology is not yet consistent. You might hear it as *Explainable AI (XAI)*, for example. Some people (like me) use these terms as synonyms, [others](https://towardsdatascience.com/shall-we-build-transparent-models-right-away-196db0eeba6c) make a distinction between them. The lingo is just not consistent yet.

This article is heavily based on Christoph Molnar's book Interpretable Machine Learning, which is available [for free online](https://christophm.github.io/interpretable-ml-book/).

<img src="img/iml-book.jpg" style="height: 200px" />

## How to Achieve Interpretability

If you want to understand and explain the behavior of your model, there are two ways to go:

### Option 1: Use interpretable models

The first option is to use simple models that are already interpretable.

One option is a linear regression model. It's always a nice first choice to run, if just as a benchmark. The regression coefficients and their p-values are a nice measure for the size and importance of each feature.

Another candidate, especially for classification, is a short decision tree. They can be visualized in an intuitive way, and most people can use it to make predictions themselves, too.

However, these models have disadvantages. Most importantly, both usually suffer from poor performance in the real world. Linear models are constrained to linear relationships, so they tend to fail whenever the real world contains nonlinear effects (think, for example, of the relationship between your age and your body height). Decision trees, on the other hand, are designed to model step functions, so they fail when the real relationship *is* in fact linear.

So there has to be a better way to achive interpretability - and luckily, there is.

### Option 2: Use black-box models and post-hoc interpretation methods




## Data and Models used


## Interpretation Methods

In this article, I'll introduce three methods for model-agnostic interpretation. Each method is solving the problems introduced in one of the three stories at the beginning.

### Permutation Feature Importance

### Partial Dependence Plots

### SHAP / Shapley Values

## Other Methods not covered here

## The Future of Interpretable Machine Learning

