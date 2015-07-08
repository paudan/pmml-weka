package net.paudan.weka.pmml.wrapper;

import org.dmg.pmml.*;
import org.jpmml.model.JAXBUtil;
import weka.classifiers.Classifier;
import weka.classifiers.RandomForestUtils;
import weka.classifiers.meta.Bagging;
import weka.core.Attribute;

import javax.xml.transform.stream.StreamSource;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import net.paudan.weka.pmml.PMMLConsumer;
import static net.paudan.weka.pmml.PMMLConversionCommons.buildInstances;
import static net.paudan.weka.pmml.PMMLConversionCommons.getClassDistribution;
import static net.paudan.weka.pmml.PMMLConversionCommons.getClassIndex;
import static net.paudan.weka.pmml.PMMLConversionCommons.getMiningModel;
import static net.paudan.weka.pmml.PMMLConversionCommons.getNodeTrainingProportion;
import net.paudan.weka.pmml.PMMLConversionException;
import weka.classifiers.RandomForestWrapper;
import weka.classifiers.RandomTreeWrapper;
import weka.core.Instances;

/**
 * A consumer that converts PMML to a {@link weka.classifiers.trees.RandomForest} instance.
 *
 * @author Ricardo Ferreira (ricardo.ferreira@feedzai.com)
 * @since 1.0.4
 */
public class RandomForestPMMLConsumer implements PMMLConsumer<RandomForestWrapper> {

    private PMML pmml;
    
    private PMML getPMML(String pmmlString) throws PMMLConversionException {
        PMML pmml = null;
        try (ByteArrayInputStream bais = new ByteArrayInputStream(pmmlString.getBytes())){
            pmml = JAXBUtil.unmarshalPMML(new StreamSource(bais));
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to load PMML. Make sure it is a valid PMML.", e);
        }
        return pmml;
    }
    
    private PMML getPMML(File file) throws PMMLConversionException {
        PMML pmml = null;
        try (FileInputStream fis = new FileInputStream(file)) {
            pmml = JAXBUtil.unmarshalPMML(new StreamSource(fis));
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to unmarshal PMML file '" + file + "'. Make sure the file is a valid PMML.", e);
        }
        return pmml;
    }

    public PMML getPMML() {
        return pmml;
    }
    
    @Override
    public RandomForestWrapper consume(String pmmlString) throws PMMLConversionException {
        pmml = getPMML(pmmlString);
        return consume(pmml);
    }

    @Override
    public RandomForestWrapper consume(File file) throws PMMLConversionException {
        pmml = getPMML(file);
        return consume(pmml);
    }

    @Override
    public RandomForestWrapper consume(PMML pmml) throws PMMLConversionException {
        MiningModel miningModel = getMiningModel(pmml);
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

        Instances instances = buildInstances(pmml.getDataDictionary());

        Classifier[] baggingClassifiers = RandomForestUtils.getBaggingClassifiers(randomForest.getM_bagger());

        for (int i = 0; i < baggingClassifiers.length; i++) {
            RandomTreeWrapper root = (RandomTreeWrapper) baggingClassifiers[i];
            buildRandomTree(root, instances, (TreeModel) segments.get(i).getModel());
        }

        return randomForest;
    }

    /**
     * Builds a new {@link weka.classifiers.trees.RandomTree Weka RandomTree} from the given {@link org.dmg.pmml.TreeModel PMML TreeModel}.
     *
     * @param root      The {@link weka.classifiers.trees.RandomTree Weka RandomTree} which is to be built.
     * @param instances The {@link weka.core.Instances} with the tree's attributes.
     * @param treeModel The {@link org.dmg.pmml.TreeModel PMML TreeModel} which is to be converted to a {@link weka.classifiers.trees.RandomTree Weka RandomTree}.
     * @return The same {@code root} instance.
     */
    private static RandomTreeWrapper buildRandomTree(RandomTreeWrapper root, Instances instances, TreeModel treeModel) {
        Instances treeInstances = new Instances(instances);
        treeInstances.setClassIndex(getClassIndex(instances, treeModel));

        root.setM_Info(treeInstances);
        root.setM_Tree(buildRandomTreeNode(root, treeModel.getNode()));

        return root;
    }

    /**
     * Builds a {@link weka.classifiers.trees.RandomTree.Tree Weka RandomTree} node
     * represented by the given {@link org.dmg.pmml.Node PMML node}.
     *
     * @param tree     The {@link weka.classifiers.trees.RandomTree Weka RandomTree} which the returned tree node is part of.
     * @param pmmlNode The {@link org.dmg.pmml.PMML PMML node} to be converted to a {@link weka.classifiers.trees.RandomTree.Tree Weka RandomTree} node.
     * @return A new {@link weka.classifiers.trees.RandomTree.Tree Weka RandomTree} node.
     */
    private static RandomTreeWrapper.TreeWrapper buildRandomTreeNode(RandomTreeWrapper tree, Node pmmlNode) {
        RandomTreeWrapper.TreeWrapper treeNode = tree.new TreeWrapper();
        //Set the class distribution.
        treeNode.setM_ClassDistribution(getClassDistribution(pmmlNode));

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
                treeNode.setM_Prop(new double[]{getNodeTrainingProportion(left), getNodeTrainingProportion(right)});
            } else if (attribute.isNominal()) {

                treeNode.setM_Successors(new RandomTreeWrapper.TreeWrapper[children.size()]);
                treeNode.setM_Prop(new double[treeNode.getM_Successors().length]);

                for (Node child : children) {
                    SimplePredicate predicate = (SimplePredicate) child.getPredicate();
                    int valueIndex = attribute.indexOfValue(predicate.getValue());
                    
                    treeNode.setM_Successor(buildRandomTreeNode(tree, child), valueIndex);
                    treeNode.setM_Prop(getNodeTrainingProportion(child), valueIndex);
                }
            } else {
                throw new RuntimeException("Attribute type not supported: " + attribute);
            }
        }

        return treeNode;
    }
}
