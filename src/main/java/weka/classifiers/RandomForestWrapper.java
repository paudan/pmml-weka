package weka.classifiers;

import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.Utils;

public class RandomForestWrapper extends RandomForest {
    
    @Override
    public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();

    m_bagger = new Bagging();

    // RandomTree implements WeightedInstancesHandler, so we can
    // represent copies using weights to achieve speed-up.
    m_bagger.setRepresentCopiesUsingWeights(true);

    RandomTree rTree = new RandomTreeWrapper();

    // set up the random tree options
    m_KValue = m_numFeatures;
    if (m_KValue < 1) {
      m_KValue = (int) Utils.log2(data.numAttributes() - 1) + 1;
    }
    rTree.setKValue(m_KValue);
    rTree.setMaxDepth(getMaxDepth());
    rTree.setDoNotCheckCapabilities(true);

    // set up the bagger and build the forest
    m_bagger.setClassifier(rTree);
    m_bagger.setSeed(m_randomSeed);
    m_bagger.setNumIterations(m_numTrees);
    m_bagger.setCalcOutOfBag(true);
    m_bagger.setNumExecutionSlots(m_numExecutionSlots);
    m_bagger.buildClassifier(data);
  }

    public Bagging getM_bagger() {
        return m_bagger;
    }

    public void setM_bagger(Bagging m_bagger) {
        this.m_bagger = m_bagger;
    }
    
    
}
