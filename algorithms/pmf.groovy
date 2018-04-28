import org.grouplens.lenskit.iterative.IterationCount
import org.grouplens.lenskit.iterative.StoppingThreshold
import org.lenskit.api.ItemScorer
import org.lenskit.mf.funksvd.FeatureCount
import org.lenskit.pf.*

bind ItemScorer to HPFItemScorer

//bind HPFModel toProvider HPFModelProvider
set StoppingThreshold to 0.000001
set IsProbabilityPrediction to false
set ConvergenceCheckFrequency to 10
set IterationCount to 200
set ItemActivityPriorShp to 0.5
set ItemWeightPriorShp to 0.5
set UserActivityPriorShp to 0.5
set UserWeightPriorShp to 0.5
set RandomSeed to System.currentTimeMillis()
set FeatureCount to 90
//              set UserActivityPriorMean to b
//              set ItemActivityPriorMean to b
