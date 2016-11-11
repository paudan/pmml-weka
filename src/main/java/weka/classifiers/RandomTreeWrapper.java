package weka.classifiers;

import java.util.Random;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class RandomTreeWrapper extends RandomTree {

    /**
    * Builds classifier.
    * 
    * @param data the data to train with
    * @throws Exception if something goes wrong or the data doesn't fit
    */
   @Override
   public void buildClassifier(Instances data) throws Exception {

     // Make sure K value is in range
     if (m_KValue > data.numAttributes() - 1) {
       m_KValue = data.numAttributes() - 1;
     }
     if (m_KValue < 1) {
       m_KValue = (int) Utils.log2(data.numAttributes() - 1) + 1;
     }

     // can classifier handle the data?
     getCapabilities().testWithFail(data);

     // remove instances with missing class
     data = new Instances(data);
     data.deleteWithMissingClass();

     // only class? -> build ZeroR model
     if (data.numAttributes() == 1) {
       System.err
         .println("Cannot build model (only class attribute present in data!), "
           + "using ZeroR model instead!");
       m_zeroR = new weka.classifiers.rules.ZeroR();
       m_zeroR.buildClassifier(data);
       return;
     } else {
       m_zeroR = null;
     }

     // Figure out appropriate datasets
     Instances train = null;
     Instances backfit = null;
     Random rand = data.getRandomNumberGenerator(m_randomSeed);
     if (m_NumFolds <= 0) {
       train = data;
     } else {
       data.randomize(rand);
       data.stratify(m_NumFolds);
       train = data.trainCV(m_NumFolds, 1, rand);
       backfit = data.testCV(m_NumFolds, 1);
     }

     // Create the attribute indices window
     int[] attIndicesWindow = new int[data.numAttributes() - 1];
     int j = 0;
     for (int i = 0; i < attIndicesWindow.length; i++) {
       if (j == data.classIndex()) {
         j++; // do not include the class
       }
       attIndicesWindow[i] = j++;
     }

     double totalWeight = 0;
     double totalSumSquared = 0;

     // Compute initial class counts
     double[] classProbs = new double[train.numClasses()];
     for (int i = 0; i < train.numInstances(); i++) {
       Instance inst = train.instance(i);
       if (data.classAttribute().isNominal()) {
         classProbs[(int) inst.classValue()] += inst.weight();
         totalWeight += inst.weight();
       } else {
         classProbs[0] += inst.classValue() * inst.weight();
         totalSumSquared += inst.classValue() * inst.classValue()
           * inst.weight();
         totalWeight += inst.weight();
       }
     }

     double trainVariance = 0;
     if (data.classAttribute().isNumeric()) {
       trainVariance = RandomTree.singleVariance(classProbs[0], totalSumSquared,
         totalWeight) / totalWeight;
       classProbs[0] /= totalWeight;
     }

     // Build tree
     m_Tree = new TreeWrapper();
     m_Info = new Instances(data, 0);
     ((TreeWrapper)m_Tree).buildTree(train, classProbs, attIndicesWindow, totalWeight, rand, 0,
       m_MinVarianceProp * trainVariance);

     // Backfit if required
     if (backfit != null) {
       m_Tree.backfitData(backfit);
     }
   }

    public Instances getM_Info() {
        return m_Info;
    }

    public TreeWrapper getM_Tree() {
        return (TreeWrapper) m_Tree;
    }

    public void setM_Tree(Tree m_Tree) {
        this.m_Tree = m_Tree;
    }

    public void setM_Info(Instances m_Info) {
        this.m_Info = m_Info;
    }
    
    

    public class TreeWrapper extends Tree {
        
        public TreeWrapper[] getM_Successors() {
            return (TreeWrapper[]) m_Successors;
        }

        public double getM_SplitPoint() {
            return m_SplitPoint;
        }

        public double[] getM_Prop() {
            return m_Prop;
        }

        public int getM_Attribute() {
            return m_Attribute;
        }

        public double[] getM_ClassDistribution() {
            return m_ClassDistribution;
        }

        public double[] getM_Distribution() {
            return m_Distribution;
        }

        public void setM_Attribute(int m_Attribute) {
            this.m_Attribute = m_Attribute;
        }

        public void setM_ClassDistribution(double[] m_ClassDistribution) {
            this.m_ClassDistribution = m_ClassDistribution;
        }

        public void setM_Successors(Tree[] m_Successors) {
            this.m_Successors = m_Successors;
        }
        
        public void setM_Successor(Tree m_Successor, int index) {
            this.m_Successors[index] = m_Successor;
        }

        public void setM_SplitPoint(double m_SplitPoint) {
            this.m_SplitPoint = m_SplitPoint;
        }

        public void setM_Prop(double[] m_Prop) {
            this.m_Prop = m_Prop;
        }
        
        public void setM_Prop(double m_Prop, int valueIndex) {
            this.m_Prop[valueIndex] = m_Prop;
        }

        public void setM_Distribution(double[] m_Distribution) {
            this.m_Distribution = m_Distribution;
        }

        /**
        * Recursively generates a tree.
        * 
        * @param data the data to work with
        * @param classProbs the class distribution
        * @param attIndicesWindow the attribute window to choose attributes from
        * @param random random number generator for choosing random attributes
        * @param depth the current depth
        * @throws Exception if generation fails
        */
        @Override
        public void buildTree(Instances data, double[] classProbs,
          int[] attIndicesWindow, double totalWeight, Random random, int depth,
          double minVariance) throws Exception {

             // Make leaf if there are no training instances
             if (data.numInstances() == 0) {
               m_Attribute = -1;
               m_ClassDistribution = null;
               m_Prop = null;

               if (data.classAttribute().isNumeric()) {
                 m_Distribution = new double[2];
               }
               return;
             }

             double priorVar = 0;
             if (data.classAttribute().isNumeric()) {

               // Compute prior variance
               double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
               for (int i = 0; i < data.numInstances(); i++) {
                 Instance inst = data.instance(i);
                 totalSum += inst.classValue() * inst.weight();
                 totalSumSquared += inst.classValue() * inst.classValue()
                   * inst.weight();
                 totalSumOfWeights += inst.weight();
               }
               priorVar = RandomTree.singleVariance(totalSum, totalSumSquared,
                 totalSumOfWeights);
             }

             // Check if node doesn't contain enough instances or is pure
             // or maximum depth reached
             if (data.classAttribute().isNominal()) {
               totalWeight = Utils.sum(classProbs);
             }
             // System.err.println("Total weight " + totalWeight);
             // double sum = Utils.sum(classProbs);
             if (totalWeight < 2 * m_MinNum ||

             // Nominal case
               (data.classAttribute().isNominal() && Utils.eq(
                 classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs)))

               ||

               // Numeric case
               (data.classAttribute().isNumeric() && priorVar / totalWeight < minVariance)

               ||

               // check tree depth
               ((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {

               // Make leaf
               m_Attribute = -1;
               m_ClassDistribution = classProbs.clone();
               if (data.classAttribute().isNumeric()) {
                 m_Distribution = new double[2];
                 m_Distribution[0] = priorVar;
                 m_Distribution[1] = totalWeight;
               }

               m_Prop = null;
               return;
             }

             // Compute class distributions and value of splitting
             // criterion for each attribute
             double val = -Double.MAX_VALUE;
             double split = -Double.MAX_VALUE;
             double[][] bestDists = null;
             double[] bestProps = null;
             int bestIndex = 0;

             // Handles to get arrays out of distribution method
             double[][] props = new double[1][0];
             double[][][] dists = new double[1][0][0];
             double[][] totalSubsetWeights = new double[data.numAttributes()][0];

             // Investigate K random attributes
             int attIndex = 0;
             int windowSize = attIndicesWindow.length;
             int k = m_KValue;
             boolean gainFound = false;
             double[] tempNumericVals = new double[data.numAttributes()];
             while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

               int chosenIndex = random.nextInt(windowSize);
               attIndex = attIndicesWindow[chosenIndex];

               // shift chosen attIndex out of window
               attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
               attIndicesWindow[windowSize - 1] = attIndex;
               windowSize--;

               double currSplit = data.classAttribute().isNominal() ? distribution(
                 props, dists, attIndex, data) : numericDistribution(props, dists,
                 attIndex, totalSubsetWeights, data, tempNumericVals);

               double currVal = data.classAttribute().isNominal() ? gain(dists[0],
                 priorVal(dists[0])) : tempNumericVals[attIndex];

               if (Utils.gr(currVal, 0)) {
                 gainFound = true;
               }

               if ((currVal > val) || ((currVal == val) && (attIndex < bestIndex))) {
                 val = currVal;
                 bestIndex = attIndex;
                 split = currSplit;
                 bestProps = props[0];
                 bestDists = dists[0];
               }
             }

             // Find best attribute
             m_Attribute = bestIndex;

             // Any useful split found?
             if (Utils.gr(val, 0)) {

               // Build subtrees
               m_SplitPoint = split;
               m_Prop = bestProps;
               Instances[] subsets = splitData(data);
               m_Successors = new TreeWrapper[bestDists.length];
               double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];

               for (int i = 0; i < bestDists.length; i++) {
                 m_Successors[i] = new TreeWrapper();
                 ((TreeWrapper)m_Successors[i]).buildTree(subsets[i], bestDists[i], attIndicesWindow,
                   data.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i],
                   random, depth + 1, minVariance);
               }

               // If all successors are non-empty, we don't need to store the class
               // distribution
               boolean emptySuccessor = false;
               for (int i = 0; i < subsets.length; i++) {
                 if (((TreeWrapper)m_Successors[i]).m_ClassDistribution == null) {
                   emptySuccessor = true;
                   break;
                 }
               }
               if (emptySuccessor) {
                 m_ClassDistribution = classProbs.clone();
               }
             } else {

               // Make leaf
               m_Attribute = -1;
               m_ClassDistribution = classProbs.clone();
               if (data.classAttribute().isNumeric()) {
                 m_Distribution = new double[2];
                 m_Distribution[0] = priorVar;
                 m_Distribution[1] = totalWeight;
               }
             }
           }
    }
    
}
