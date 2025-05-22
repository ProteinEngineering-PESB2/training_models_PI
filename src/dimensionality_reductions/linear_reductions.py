from sklearn.decomposition import (IncrementalPCA,
                                   PCA, SparsePCA, MiniBatchSparsePCA)

from dimensionality_reductions.reductionMethods import Reductions

class LinearReduction(Reductions):

    def __init__(
            self,
            dataset=None):
        
        super().__init__(
            dataset = dataset
        )
    
    def applyPCA(
            self,
            n_components=2,
            svd_solver="auto",
            random_state=42):

        pca_instance = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=random_state
        )

        pca_instance.fit(self.dataset.values)
        transform_values = pca_instance.transform(self.dataset.values)

        return pca_instance, self.generateDatasetPostReduction(transform_values, n_components)
    
    def applyIncrementalPCA(
            self,
            n_components=2):

        incremental_instance = IncrementalPCA(
            n_components=n_components,
        )

        incremental_instance.fit(self.dataset.values)
        transform_values = incremental_instance.transform(self.dataset.values)

        return incremental_instance, self.generateDatasetPostReduction(transform_values, n_components)
    
    def applySparsePCA(
            self,
            n_components=2,
            method="lars",
            random_state=42):

        sparsePCA_instance = SparsePCA(
            n_components=n_components,
            method=method,
            random_state=random_state
        )

        sparsePCA_instance.fit(self.dataset.values)
        transform_values = sparsePCA_instance.transform(self.dataset.values)

        return sparsePCA_instance, self.generateDatasetPostReduction(transform_values, n_components)
    
    def applyMiniBatchSparsePCA(
            self,
            n_components=2,
            method="lars",
            random_state=42,
            max_iter=1000):

        mini_batch_sparsePCA_instance = MiniBatchSparsePCA(
            n_components=n_components,
            method=method,
            random_state=random_state,
            max_iter=max_iter
        )

        mini_batch_sparsePCA_instance.fit(self.dataset.values)
        transform_values = mini_batch_sparsePCA_instance.transform(self.dataset.values)

        return mini_batch_sparsePCA_instance, self.generateDatasetPostReduction(transform_values, n_components)
