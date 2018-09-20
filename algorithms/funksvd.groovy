import org.lenskit.bias.*
import org.lenskit.mf.funksvd.*
import org.grouplens.lenskit.iterative.*

bind ItemScorer to FunkSVDItemScorer
bind BiasModel to UserItemBiasModel

set BiasDamping to 5
set FeatureCount to 15
set LearningRate to 0.001
set IterationCount to 125