# nanomlp

# Core Idea

-> the central question in machine learning  

**How does changing a parameter affect the final output?**

---

## Example Setup

-> if parameters are given

```
a = -4
b = 2
```

-> imagine some complicated function

$$
g = f(a,b)
$$

---

## Question We Ask

If I slightly change **a** → how does **g** change?  

If I slightly change **b** → how does **g** change?

---

## Sensitivity

this sensitivity is called **gradient**

$$
\frac{\partial g}{\partial a}
$$

$$
\frac{\partial g}{\partial b}
$$

This idea powers **machine learning training**.

---

# What a Derivative Means

-> derivative measures **slope**

$$
\frac{f(x+h)-f(x)}{h}
$$

where

```
h → very small number
```

Interpretation

```
small change in x
→ how much f(x) changes
```

---

## Example Function

$$
f(x) = 3x^2 - 4x + 5
$$

At

```
x = 3
```

Slope = **14**

Meaning

```
if x increases slightly
f(x) increases 14 times faster
```

---

# Neural Networks

-> neural networks are **just big math expressions**

```
prediction = model(x, weights)
```

Inside the model

```
((x1*w1 + x2*w2) + b)
↓
activation
↓
more layers
↓
output
```

Operations used

```
+
*
exp
tanh
```

Nothing mystical.

---

# Training Goal

Training means

```
adjust weights
so loss decreases
```

For that we compute

$$
\frac{\partial loss}{\partial weights}
$$

This is where **backpropagation** comes from.

---

# Computation Graph

Instead of one giant equation we break it into steps.

Example values

```
a = 2
b = -3
c = 10
```

Expression

```
e = a * b
d = e + c
```

Graph

```
a ----\
       * ---- e ----\
b ----/              +
                     \
c -------------------- d
```

Each node stores

```
value
operation
parents
gradient
```

This structure is called a **computation graph**.

---

# Forward Pass

Forward pass = normal calculation.

Given

```
a = 2
b = -3
c = 10
```

Compute

```
e = a * b = -6
d = e + c = 4
```

Final output

```
d = 4
```

---

# Backward Pass

Now we compute gradients.

How sensitive is **d** to each variable?

```
∂d/∂a
∂d/∂b
∂d/∂c
```

---

## Step 1

```
d = e + c
```

Derivatives

```
∂d/∂e = 1
∂d/∂c = 1
```

Meaning

```
increase e → d increases equally
increase c → d increases equally
```

---

## Step 2

```
e = a * b
```

Derivatives

```
∂e/∂a = b
∂e/∂b = a
```

So

```
∂e/∂a = -3
∂e/∂b = 2
```

---

# Chain Rule

If

```
z depends on y
y depends on x
```

Then

$$
\frac{dz}{dx} =
\frac{dz}{dy} \times \frac{dy}{dx}
$$

Interpretation

```
effect of x on z
=
effect of x on y
×
effect of y on z
```

Influence flows through the graph.

---

# Apply Chain Rule

We know

```
∂d/∂e = 1
∂e/∂a = -3
```

So

```
∂d/∂a = 1 * (-3) = -3
```

Similarly

```
∂d/∂b = 1 * 2 = 2
```

and

```
∂d/∂c = 1
```

These values are the **gradients**.

---

# Gradient Descent

Suppose

```
d = loss
```

Parameters

```
a b c
```

Update rule

```
parameter = parameter - learning_rate * gradient
```

Example

```
a = a - 0.01 * (-3)
```

Meaning

```
a increases
```

Because increasing **a** reduces the loss.

as both equation shows negative it depend on a situation 
what if gradient is positive then we have to decrease the a such to decrease the loss 
and if gradient is negative then we have to increase the a such to decrease the loss 
but in both case we just subtract cause it same 


---

# Neuron

A neuron computes

```
x1*w1 + x2*w2 + ... + bias
```

Then activation

```
tanh()
```

Full neuron equation

$$
output = \tanh(w_1x_1 + w_2x_2 + b)
$$

Activation introduces **non-linearity**.

Without it networks collapse into linear regression.

---

# Layer

A layer = multiple neurons.

Example

```
3 inputs → 4 neurons
```

Each neuron produces one output.

---

# Multi Layer Perceptron

Layers stacked together

```
Input → Hidden → Hidden → Output
```

Example architecture

```
3 → 4 → 4 → 1
```

Meaning

```
3 inputs
4 neurons
4 neurons
1 output
```

---

# Loss Function

Prediction compared with target.

$$
(y_{pred} - y_{true})^2
$$

Goal

```
minimize loss
```

---

# Training Loop

Training follows a simple cycle.

```
1 forward pass
2 compute loss
3 backward pass
4 update weights
```

Example loop

```python
for step in range(100):

    y_pred = model(x)

    loss = compute_loss(y_pred, y)

    loss.backward()

    for p in parameters:
        p.data -= lr * p.grad

    zero_grad()
```

This is **neural network training**.

---

# Big Picture

Neural networks are simply

```
input
↓
giant math expression
↓
loss
↓
backpropagation
↓
parameter updates
```

Frameworks like

```
PyTorch
TensorFlow
JAX
```

are basically **optimized autograd engines**.


# COMPLETED A MICROGRAD HEAR ONLY

