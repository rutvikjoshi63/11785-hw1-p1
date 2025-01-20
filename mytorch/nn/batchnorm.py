import numpy as np


class BatchNorm1d:
    """
    Create your own mytorch.nn.BatchNorm1d!
    Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d class.
    Hint: Read all the expressions given in the writeup and be CAREFUL to re-check your code.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d forward.
        Note: The eval parameter indicate whether it's training phase or the inference phase of the problem.
        Check the values you need to recompute when eval = False.
        """
        self.Z = Z
        self.N = np.shape(Z)[0]  # TODO: Calculate batch size
        self.M = np.mean(self.Z, axis=0, keepdims=True)  # TODO: Calculate mini-batch mean (N,num_features)
        self.V = np.var(self.Z, axis=0, keepdims=True)  # TODO: Calculate mini-batch variance (N,num_features)
        # print("Z", self.Z, "\n M", self.M, "\n V", self.V)
        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M)/ np.sqrt(self.V + self.eps)  # TODO: Calculate the normalized input Ẑ
            self.BZ = self.BW * self.NZ + self.Bb  # TODO: Calculate the scaled and shifted for the normalized input Ẑ

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M # TODO: Calculate running mean
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V  # TODO: Calculate running variance
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M)/np.sqrt(self.running_V + self.eps)  # TODO: Calculate the normalized input Ẑ using the running average for mean and variance
            self.BZ = self.BW * self.NZ + self.Bb  # TODO: Calculate the scaled and shifted for the normalized input Ẑ

        return self.BZ #NZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d backward.
        """
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)  # TODO: Sum over the batch dimension. # , axis=0, keepdims=True
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)  # TODO: Scale gradient of loss wrt BatchNorm transformation by normalized input NZ.

        dLdNZ = dLdBZ * self.BW  # TODO: Scale gradient of loss wrt BatchNorm transformation output by gamma (scaling parameter).

        sig_eps = np.power((self.V + self.eps),-1.5)
        dLdV = -0.5 * np.sum(dLdNZ * (self.Z - self.M) * sig_eps, axis=0, keepdims=True)  # TODO: Compute gradient of loss backprop through variance calculation.
        dNZdM = -np.power((self.V + self.eps),-0.5) - 0.5*(self.Z - self.M) * np.power((self.V + self.eps),-1.5) * (-2 / self.N * np.sum(self.Z - self.M)) # TODO: Compute derivative of normalized input with respect to mean.
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)  # TODO: Compute gradient of loss with respect to mean.

        dLdZ = dLdNZ * np.power((self.V + self.eps),-0.5) + dLdV * (2/self.N*(self.Z - self.M)) + dLdM / self.N # TODO: Compute gradient of loss with respect to the input.
        # print("dLdBZ", dLdBZ, "\n self.dLdBb", self.dLdBb,"\n self.NZ", self.NZ,"\n self.dLdBW", self.dLdBW, "\n"+"-"*30)
        # print("\n dLdV", dLdV,"\n dNZdM", dNZdM,"\n dLdNZ", dLdNZ, "\n dLdM",dLdM, "\n -", "-"*30)
        
        return dLdZ  # TODO - What should be the return value?
