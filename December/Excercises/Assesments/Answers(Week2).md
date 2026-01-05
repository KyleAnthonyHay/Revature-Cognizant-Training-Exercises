### Q1:

A visualization tool that helps you see training, metrics, debugging and callbacks

### Q2:

An autoEncoder is a type of Neural network designed to learn how to compress and reconstruct user input. 

Encoder: compressesinput by putting it into a smaller latent space
Bottleneck: :  Compressed prepresentation of user input(the layer that stores the reduced form of the data).
decoder: attempts to reconstructs the original input 

### Q3: What is backpropagation and how does it enable neural networks to learn?

Backpropagation is an algorithm that calculates the gradient of the loss function for each weight in the neural network. 

It does a forward pass through the layers making predictions on the way. It calculates the loss using the loss function at the end. It then uses the chain rule to compute the gradients that reduce the error.

Finally, it updates the weights in that direction.

### Q4:

SGD: updates weights using a fixed learning rate
RMSprop (adaptive, good convergence): updates weights individually based on how recent gradients have behaved using average magnitude
Adam (adaptive, best convergence): updates weights individually like RMSprop but also incorporates momentum, which analyzes magnitude and direction


### Q5: What is batch normalization and why is it used?
Its a method of standardizes the inputs of each layer to reduce internal covariate shift, which helps the network train faster and more stable. It also includes learnable parameters to scale and shift the outputs


normalizes the inputs of each layer to reduce something called internal covariate shift. This basically means it keeps the distribution of inputs consistent, which helps the network train faster and more reliably. It also adds a bit of stability and can let you use a larger learning rate. So in short, it normalizes, speeds up training, and makes things more stable.