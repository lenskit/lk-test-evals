import org.grouplens.lenskit.iterative.IterationCount
import org.lenskit.api.ItemScorer
import org.lenskit.bias.BiasModel
import org.lenskit.bias.ZeroBiasModel
import org.lenskit.mf.BiasedMFItemScorer
import org.lenskit.mf.MFModel
import org.lenskit.mf.bpr.BPRMFModelProvider
import org.lenskit.mf.bpr.BPRTrainingSampler
import org.lenskit.mf.bpr.ImplicitTrainingSampler
import org.lenskit.mf.bpr.BatchSize
import org.lenskit.mf.funksvd.FeatureCount

bind ItemScorer to BiasedMFItemScorer
bind BiasModel to ZeroBiasModel
bind MFModel toProvider BPRMFModelProvider
bind BPRTrainingSampler to ImplicitTrainingSampler

set FeatureCount to 25
set IterationCount to 1000
set BatchSize to 10000