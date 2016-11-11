package net.paudan.weka.pmml;

import org.jpmml.model.JAXBUtil;
import weka.classifiers.Classifier;
import weka.classifiers.RandomForestUtils;
import weka.classifiers.meta.Bagging;
import weka.core.Attribute;

import javax.xml.transform.stream.StreamSource;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.lang.reflect.Field;
import java.util.List;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.Node;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.Segment;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.TreeModel;
import weka.classifiers.RandomForestWrapper;
import weka.classifiers.RandomTreeWrapper;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * A consumer that converts PMML to a {@link weka.classifiers.trees.RandomForest} instance.
 *
 * @author Paulius Danenas (danpaulius@gmail.com), based on code by Ricardo Ferreira (ricardo.ferreira@feedzai.com)
 */
public class RandomForestPMMLConsumer implements PMMLConsumer<RandomForest> {

    private PMML pmml;
    
    private PMML getPMML(String pmmlString) throws PMMLConversionException {
        PMML pmml = null;
        try (ByteArrayInputStream bais = new ByteArrayInputStream(pmmlString.getBytes())){
            pmml = JAXBUtil.unmarshalPMML(new StreamSource(bais));
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to load classifier from PMML string. Make sure it is a valid PMML.", e);
        }
        return pmml;
    }
    
    private PMML getPMML(File file) throws PMMLConversionException {
        PMML pmml = null;
        try (FileInputStream fis = new FileInputStream(file)) {
            pmml = JAXBUtil.unmarshalPMML(new StreamSource(fis));
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to load classifier from file '" + file + "'. Make sure the file is a valid PMML.", e);
        }
        return pmml;
    }

    public PMML getPMML() {
        return pmml;
    }
    
    @Override
    public RandomForest consume(String pmmlString) throws PMMLConversionException {
        pmml = getPMML(pmmlString);
        return consume(pmml);
    }

    @Override
    public RandomForest consume(File file) throws PMMLConversionException {
        pmml = getPMML(file);
        return consume(pmml);
    }

    @Override
    public RandomForest consume(PMML pmml) throws PMMLConversionException {
        try {
            RandomForestWrapper wrapper = consumeWrapper(pmml);
            RandomForest randomForest = new RandomForest();
            
            Field field = RandomForest.class.getDeclaredField("m_bagger");
            field.setAccessible(true);
            field.set(randomForest, wrapper.getM_bagger());
            
            return randomForest;
        } catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException ex) {
            throw new PMMLConversionException(ex);
        }
    }
    
    public RandomForestWrapper consumeWrapper(PMML pmml) throws PMMLConversionException {
        MiningModel miningModel = PMMLUtils.getMiningModel(pmml);
        List<Segment> segments = miningModel.getSegmentation().getSegments();

        int m_numTrees = segments.size();

        RandomForestWrapper randomForest = new RandomForestWrapper();
        Bagging bagger = new Bagging();
        bagger.setNumIterations(m_numTrees);
        bagger.setClassifier(new RandomTreeWrapper());
        randomForest.setM_bagger(bagger);

        try {
            RandomForestUtils.setupBaggingClassifiers(randomForest.getM_bagger());
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to initialize bagging classifiers.", e);
        }

        Instances instances = PMMLUtils.buildInstances(pmml.getDataDictionary());

        Classifier[] baggingClassifiers = RandomForestUtils.getBaggingClassifiers(randomForest.getM_bagger());

        for (int i = 0; i < baggingClassifiers.length; i++) {
            RandomTreeWrapper root = (RandomTreeWrapper) baggingClassifiers[i];
            buildRandomTree(root, instances, (TreeModel) segments.get(i).getModel());
        }

        return randomForest;
    }

    /**
     * Builds a new {@link weka.classifiers.trees.RandomTree Weka RandomTree} from the given {@link org.dmg.pmml.TreeModel PMML TreeModel}.
     */
    private static RandomTreeWrapper buildRandomTree(RandomTreeWrapper root, Instances instances, TreeModel treeModel) {
        Instances treeInstances = new Instances(instances);
        treeInstances.setClassIndex(PMMLUtils.getClassIndex(instances, treeModel));

        root.setM_Info(treeInstances);
        root.setM_Tree(buildRandomTreeNode(root, treeModel.getNode()));

        return root;
    }

    /**
     * Builds a {@link weka.classifiers.trees.RandomTree.Tree Weka RandomTree} node
     * represented by the given {@link org.dmg.pmml.Node PMML node}.
     */
    private static RandomTreeWrapper.TreeWrapper buildRandomTreeNode(RandomTreeWrapper tree, Node pmmlNode) {
        RandomTreeWrapper.TreeWrapper treeNode = tree.new TreeWrapper();
        //Set the class distribution.
        treeNode.setM_ClassDistribution(PMMLUtils.getClassDistribution(pmmlNode));

        Instances instances = tree.getM_Info();

        boolean isLeaf = pmmlNode.getNodes().isEmpty();

        if (!isLeaf) {
            List<Node> children = pmmlNode.getNodes();

            String attributeName = ((SimplePredicate) children.get(0).getPredicate()).getField().getValue();
            Attribute attribute = instances.attribute(attributeName);

            treeNode.setM_Attribute(attribute.index());

            if (attribute.isNumeric()) {

                assert children.size() == 2 : "Numeric attributes must have exactly 2 children";

                Node left = children.get(0);
                Node right = children.get(1);

                Predicate leftPredicate = left.getPredicate();
                Predicate rightPredicate = right.getPredicate();

                assert leftPredicate instanceof SimplePredicate && 
                       leftPredicate.getClass().equals(rightPredicate.getClass()) : 
                       "Numeric attribute's nodes must have the same simple predicate.";

                double splitPoint = Double.valueOf(((SimplePredicate) leftPredicate).getValue());

                treeNode.setM_SplitPoint(splitPoint);
                treeNode.setM_Successors(new RandomTreeWrapper.TreeWrapper[]{buildRandomTreeNode(tree, left), buildRandomTreeNode(tree, right)});
                treeNode.setM_Prop(new double[]{PMMLUtils.getNodeTrainingProportion(left), 
                    PMMLUtils.getNodeTrainingProportion(right)});
            } else if (attribute.isNominal()) {

                treeNode.setM_Successors(new RandomTreeWrapper.TreeWrapper[children.size()]);
                treeNode.setM_Prop(new double[treeNode.getM_Successors().length]);

                for (Node child : children) {
                    SimplePredicate predicate = (SimplePredicate) child.getPredicate();
                    int valueIndex = attribute.indexOfValue(predicate.getValue());
                    
                    treeNode.setM_Successor(buildRandomTreeNode(tree, child), valueIndex);
                    treeNode.setM_Prop(PMMLUtils.getNodeTrainingProportion(child), valueIndex);
                }
            } else {
                throw new RuntimeException("Attribute type not supported: " + attribute);
            }
        }

        return treeNode;
    }
}
