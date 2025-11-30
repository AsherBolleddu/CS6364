import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        converged = False

        while not converged:
            converged = True  # Assume we've converged until we find a mistake

            for x, y in dataset.iterate_once(batch_size):
                # Get the true label as a scalar
                y_true = nn.as_scalar(y)

                # Get our prediction
                prediction = self.get_prediction(x)

                # If we made a mistake, update weights
                if prediction != y_true:
                    converged = False
                    # Update rule: w = w + y * x
                    # direction is x, multiplier is y_true
                    self.w.update(x, y_true)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Architecture: input(1) -> hidden(512) -> output(1)
        # Using suggested hyperparameters from the project description
        self.hidden_size = 512
        self.learning_rate = 0.05
        self.batch_size = 200

        # First layer: input (1) -> hidden (512)
        self.W1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        # Second layer: hidden (512) -> output (1)
        self.W2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # First layer: x * W1 + b1, then ReLU
        z1 = nn.Linear(x, self.W1)
        z1_bias = nn.AddBias(z1, self.b1)
        h1 = nn.ReLU(z1_bias)

        # Second layer: h1 * W2 + b2 (no ReLU at output - we need negative values for sin)
        z2 = nn.Linear(h1, self.W2)
        output = nn.AddBias(z2, self.b2)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        loss_threshold = 0.02

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # Compute loss
                loss = self.get_loss(x, y)

                # Compute gradients for all parameters
                gradients = nn.gradients(
                    loss, [self.W1, self.b1, self.W2, self.b2])
                grad_W1, grad_b1, grad_W2, grad_b2 = gradients

                # Update parameters using gradient descent
                # w = w - learning_rate * gradient
                self.W1.update(grad_W1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.W2.update(grad_W2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)

            # Check total loss after each epoch
            total_loss = nn.as_scalar(self.get_loss(
                nn.Constant(dataset.x), nn.Constant(dataset.y)))
            if total_loss < loss_threshold:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Architecture: input(784) -> hidden(200) -> output(10)
        # Using suggested hyperparameters from the project description
        self.hidden_size = 200
        self.learning_rate = 0.5
        self.batch_size = 100

        # First layer: input (784) -> hidden (200)
        self.W1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        # Second layer: hidden (200) -> output (10)
        self.W2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # First layer: x * W1 + b1, then ReLU
        z1 = nn.Linear(x, self.W1)
        z1_bias = nn.AddBias(z1, self.b1)
        h1 = nn.ReLU(z1_bias)

        # Second layer: h1 * W2 + b2 (no ReLU - output is logits for softmax)
        z2 = nn.Linear(h1, self.W2)
        output = nn.AddBias(z2, self.b2)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        accuracy_threshold = 0.975

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # Compute loss
                loss = self.get_loss(x, y)

                # Compute gradients for all parameters
                gradients = nn.gradients(
                    loss, [self.W1, self.b1, self.W2, self.b2])
                grad_W1, grad_b1, grad_W2, grad_b2 = gradients

                # Update parameters using gradient descent
                self.W1.update(grad_W1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.W2.update(grad_W2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)

            # Check validation accuracy after each epoch
            validation_accuracy = dataset.get_validation_accuracy()
            if validation_accuracy >= accuracy_threshold:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Hyperparameters - increased hidden size for more capacity
        self.hidden_size = 300
        self.learning_rate = 0.08
        self.batch_size = 64

        # RNN parameters
        # W_x: transforms input character to hidden state
        self.W_x = nn.Parameter(self.num_chars, self.hidden_size)
        self.b_x = nn.Parameter(1, self.hidden_size)

        # W_hidden: transforms previous hidden state (used for subsequent characters)
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)

        # Output layer with an additional hidden layer for more expressiveness
        self.W_out1 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_out1 = nn.Parameter(1, self.hidden_size)
        self.W_out2 = nn.Parameter(self.hidden_size, len(self.languages))
        self.b_out2 = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        h = None

        for i, x in enumerate(xs):
            # Transform input character
            z_x = nn.Linear(x, self.W_x)

            if i == 0:
                # First character: h = ReLU(x * W_x + b_x)
                z = nn.AddBias(z_x, self.b_x)
            else:
                # Subsequent characters: h = ReLU(x * W_x + h * W_hidden + b_x)
                z_h = nn.Linear(h, self.W_hidden)
                z = nn.Add(z_x, z_h)
                z = nn.AddBias(z, self.b_x)

            # Apply ReLU activation
            h = nn.ReLU(z)

        # Output layer with additional hidden layer for more expressiveness
        out1 = nn.Linear(h, self.W_out1)
        out1 = nn.AddBias(out1, self.b_out1)
        out1 = nn.ReLU(out1)

        output = nn.Linear(out1, self.W_out2)
        output = nn.AddBias(output, self.b_out2)

        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        accuracy_threshold = 0.85

        while True:
            for xs, y in dataset.iterate_once(self.batch_size):
                # Compute loss
                loss = self.get_loss(xs, y)

                # Compute gradients for all parameters
                params = [self.W_x, self.b_x, self.W_hidden,
                          self.W_out1, self.b_out1, self.W_out2, self.b_out2]
                gradients = nn.gradients(loss, params)

                # Update parameters using gradient descent
                for param, grad in zip(params, gradients):
                    param.update(grad, -self.learning_rate)

            # Check validation accuracy after each epoch
            validation_accuracy = dataset.get_validation_accuracy()
            if validation_accuracy >= accuracy_threshold:
                break
