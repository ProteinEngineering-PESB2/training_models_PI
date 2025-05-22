from dimensionality_reductions.reductionMethods import Reductions

from sklearn.manifold import (TSNE, Isomap, LocallyLinearEmbedding, 
                              MDS, SpectralEmbedding)
import umap.umap_ as umap

class NonLinearReductions(Reductions):

    def __init__(
            self,
            dataset=None):
        
        super().__init__(
            dataset = dataset
        )
    
    def applyTSNE(
            self,
            n_components=2, 
            learning_rate="auto", 
            init="pca", 
            perplexity=3, 
            random_state=42):

        transformTSNE =  TSNE(
            n_components=n_components, 
            learning_rate=learning_rate, 
            init=init, 
            perplexity=perplexity, 
            random_state=random_state
        ).fit_transform(self.dataset.values)

        return self.generateDatasetPostReduction(transformTSNE, n_components)
    
    def applyIsomap(self):

        isomap_values = Isomap().fit_transform(self.dataset.values)
        return self.generateDatasetPostReduction(isomap_values, isomap_values.shape[1])
    
    def applyMDE(
            self,
            n_components=2, 
            random_state=42):

        mds_values = MDS(
            n_components=n_components, 
            random_state=random_state).fit_transform(self.dataset.values)
        
        return self.generateDatasetPostReduction(mds_values, n_components)
    
    def applyLLE(
            self,
            n_components=2, 
            random_state=42):
        
        lle_values = LocallyLinearEmbedding(
            n_components=n_components, 
            random_state=random_state).fit_transform(self.dataset.values)
        
        return self.generateDatasetPostReduction(lle_values, n_components)
    
    def applySpectral(
            self,
            n_components=2):

        spectral_values = SpectralEmbedding(n_components=n_components).fit_transform(self.dataset.values)

        return self.generateDatasetPostReduction(spectral_values, n_components)
    
    def applyUMAP(
            self,
            random_state=42):

        umap_values = umap.UMAP(random_state=random_state).fit_transform(self.dataset.values)
        return self.generateDatasetPostReduction(umap_values, umap_values.shape[1])

