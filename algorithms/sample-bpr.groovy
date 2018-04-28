import org.apache.commons.math3.distribution.ExponentialDistribution
import org.grouplens.lenskit.iterative.IterationCount
import org.grouplens.lenskit.iterative.LearningRate
import org.grouplens.lenskit.iterative.RegularizationTerm
import org.lenskit.api.ItemScorer
import org.lenskit.bias.BiasModel
import org.lenskit.bias.ZeroBiasModel
import org.lenskit.mf.BiasedMFItemScorer
import org.lenskit.mf.MFModel
import org.lenskit.mf.bpr.BPRMFModelProvider
import org.lenskit.mf.bpr.BPRTrainingSampler
import org.lenskit.mf.bpr.BatchSize
import org.lenskit.mf.bpr.ImplicitTrainingSampler
import org.lenskit.mf.funksvd.FeatureCount

bind ItemScorer to BiasedMFItemScorer
bind BiasModel to ZeroBiasModel
bind MFModel toProvider BPRMFModelProvider
bind BPRTrainingSampler to ImplicitTrainingSampler

set FeatureCount to 25
set IterationCount to 100
set BatchSize to 10000

def lrGen = new ExponentialDistribution(0.5e-3)
def rgGen = new ExponentialDistribution(1.0e-2)

for (int i = 0; i < 100; i++) {
    def lr = lrGen.sample() + 1.0e-6
    def rg = rgGen.sample()
    algorithm("BPR") {
        set LearningRate to lr
        attributes["Rate"] = lr
        set RegularizationTerm to rg
        attributes["Reg"] = rg
    }
}
