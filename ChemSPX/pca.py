import numpy as np
from sklearn.decomposition import PCA
from ChemSPX.input_parser import InputParser


class PCA:
    """
    Class for dimentionality reduction with PCA
    """

    def __init__(self, data_file):

        self.inp = InputParser(input)
        self.indict = self.inp.get()
        self.ref_data_file = self.indict["in_file"]
        self.data_file = data_file
        self.pca_comp = self.indict["pca_n_components"]
        self.ref_data = np.genfromtxt(self.ref_data_file, delimiter=",", dtype=float)
        self.data = np.genfromtxt(self.data_file, delimiter=",", dtype=float)
        self.ref_data = self.ref_data[:, :-1]
        self.all_data = np.vstack((self.ref_data, self.data))
        self.ref_data_idx = len(self.ref_data)
        self.data_idx = len(self.data)

        self.index = np.zeros(self.ref_data_idx + self.data_idx)
        for i in range(len(self.data_idx)):
            self.index[self.ref_data_idx + i] = 1

    def reduce(self):
        pca = PCA(n_components=self.pca_comp)
        principalComponents = pca.fit_transform(self.all_data)
        principalComponents = np.append(principalComponents, self.index, axis=1)
        f = open(f'{self.indict["out_dir"]}/pca.csv', "a")
        np.savetxt(f, principalComponents, delimiter=",")
