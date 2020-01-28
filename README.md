# NeuralNetworks
A python library mainly designed for my own use. Eventual implementation of various types of neural networks, supervised and unsupervised learning and other machine learning features from scratch.

Currently implemented:
 - feed-forward neural networks
 - GUI for said networks
 - application to a language model as a word embedder.
 
 Word embedder:
 A rudimentary dictionary vector with binary states for each word gets translated into a n-vector which encodes substantial information about the word itself. Vector algebra then yields sensible results. i.e. a vector encoding "King" minus a vector encoding "Man" plus a vector encoding "Woman" yields the next closest vector in the dictionary as "Queen". This is done using a relatively small FFNN and backpropagation. The front and end layer have dimensions of the original dict vector. The middle layer has n nodes and  will later be used as the more sophisticated vector. The model is trained by matching an input word wih random samples from its neigbourhood in training data texts. This allows words which appear in similiar contexts in sentences to be turned into similiar vectors. "Cat" and "Dog" for example would both be found commonly in the neighbourhood of words like "Pet" or "Cuddle".
 
 Current Problems:
  - approach limited by missing optimisation.
  
 Planned Fix:
  - use of the numba library to access the cuda framework and use GPU to make calculations.
