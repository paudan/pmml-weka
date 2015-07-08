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
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import static net.paudan.weka.pmml.PMMLConversionCommons.buildInstances;
import static net.paudan.weka.pmml.PMMLConversionCommons.getClassDistribution;
import static net.paudan.weka.pmml.PMMLConversionCommons.getClassIndex;
import static net.paudan.weka.pmml.PMMLConversionCommons.getMiningModel;
import static net.paudan.weka.pmml.PMMLConversionCommons.getNodeTrainingProportion;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.Node;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.Segment;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.TreeModel;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
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
            MiningModel miningModel = getMiningModel(pmml);
            List<Segment> segments = miningModel.getSegmentation().getSegments();
            
            int m_numTrees = segments.size();
            
            RandomForest randomForest = new RandomForest();
            Bagging bagger = new Bagging();
            bagger.setNumIterations(m_numTrees);
            bagger.setClassifier(new RandomTree());
            
            Field field = RandomForest.class.getDeclaredField("m_bagger");
            field.setAccessible(true);
            field.set(randomForest, bagger);
            
            try {
                RandomForestUtils.setupBaggingClassifiers(bagger);
            } catch (Exception e) {
                throw new PMMLConversionException("Failed to initialize bagging classifiers.", e);
            }
            
            Instances instances = buildInstances(pmml.getDataDictionary());
            Classifier[] baggingClassifiers = RandomForestUtils.getBaggingClassifiers(bagger);
            for (int i = 0; i < baggingClassifiers.length; i++) {
                RandomTree root = (RandomTree) baggingClassifiers[i];
                buildRandomTree(root, instances, (TreeModel) segments.get(i).getModel());
            }
            return randomForest;
        } catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException |
                ClassNotFoundException | NoSuchMethodException | InstantiationException | InvocationTargetException ex) {
            throw new PMMLConversionException(ex);
        } 
    }

    /**
     * Builds a new {@link weka.classifiers.trees.RandomTree Weka RandomTree} from the given {@link org.dmg.pmml.TreeModel PMML TreeModel}.
     *
     * @param root      The {@link weka.classifiers.trees.RandomTree Weka RandomTree} which is to be built.
     * @param instances The {@link weka.core.Instances} with the tree's attributes.
     * @param treeModel The {@link org.dmg.pmml.TreeModel PMML TreeModel} which is to be converted to a {@link weka.classifiers.trees.RandomTree Weka RandomTree}.
     * @return The same {@code root} instance.
     */
    private static RandomTree buildRandomTree(RandomTree root, Instances instances, TreeModel treeModel) 
            throws NoSuchFieldException, IllegalArgumentException, IllegalAccessException, ClassNotFoundException, 
            NoSuchMethodException, InstantiationException, InvocationTargetException {
        Instances treeInstances = new Instances(instances);
        treeInstances.setClassIndex(getClassIndex(instances, treeModel));

        Field field = RandomTree.class.getDeclaredField("m_Info");
        field.setAccessible(true);
        field.set(root, treeInstances);
        field = RandomTree.class.getDeclaredField("m_Tree");
        field.setAccessible(true);
        field.set(root, buildRandomTreeNode(root, treeModel.getNode()));

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
    private static Object buildRandomTreeNode(RandomTree tree, Node pmmlNode) 
            throws ClassNotFoundException, NoSuchMethodException, InstantiationException, 
            IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchFieldException {
        
        RandomTree treeObj = new RandomTree();
        Class<?> treeClass= Class.forName("weka.classifiers.trees.RandomTree$Tree");
        Constructor<?> constructor = treeClass.getDeclaredConstructor(RandomTree.class);
        constructor.setAccessible(true);
        Object treeNode = constructor.newInstance(treeObj);
        
        Field field = treeClass.getDeclaredField("m_ClassDistribution");
        field.setAccessible(true);
        double [] distribution = getClassDistribution(pmmlNode);
        if (distribution != null && distribution.length > 0) {
            Object m_ClassDistribution = Array.newInstance(double.class, distribution.length);
            for (int i = 0; i < distribution.length; i++)
                 Array.set(m_ClassDistribution, i, distribution[i]);
            field.set(treeNode, m_ClassDistribution);
        }

        field = RandomTree.class.getDeclaredField("m_Info");
        field.setAccessible(true);
        Instances instances = Instances.class.cast(field.get(tree));

        boolean isLeaf = pmmlNode.getNodes().isEmpty();

        if (!isLeaf) {
            List<Node> children = pmmlNode.getNodes();

            String attributeName = ((SimplePredicate) children.get(0).getPredicate()).getField().getValue();
            Attribute attribute = instances.attribute(attributeName);

            field = treeClass.getDeclaredField("m_Attribute");
            field.setAccessible(true);
            field.setInt(treeNode, attribute.index());

            if (attribute.isNumeric()) {

                assert children.size() == 2 : "Numeric attributes must have exactly 2 children";

                Node left = children.get(0);
                Node right = children.get(1);

                Predicate leftPredicate = left.getPredicate();
                Predicate rightPredicate = right.getPredicate();

                assert leftPredicate instanceof SimplePredicate && 
                       leftPredicate.getClass().equals(rightPredicate.getClass()) : 
                       "Numeric attribute's nodes must have the same simple predicate";

                double splitPoint = Double.valueOf(((SimplePredicate) leftPredicate).getValue());

                field = treeClass.getDeclaredField("m_SplitPoint");
                field.setAccessible(true);
                field.set(treeNode, splitPoint);
                
                field = treeClass.getDeclaredField("m_Successors");
                field.setAccessible(true);
                Object m_Successors = Array.newInstance(treeClass, 2);
                Array.set(m_Successors, 0, buildRandomTreeNode(tree, left) );
                Array.set(m_Successors, 1, buildRandomTreeNode(tree, right) );
                field.set(treeNode, m_Successors);
                
                field = treeClass.getDeclaredField("m_Prop");
                field.setAccessible(true);
                Object m_Props = Array.newInstance(double.class, 2);
                Array.setDouble(m_Props, 0, getNodeTrainingProportion(left));
                Array.setDouble(m_Props, 1, getNodeTrainingProportion(right));
                field.set(treeNode, m_Props);
                
            } else if (attribute.isNominal()) {
 
                Object m_Successors = Array.newInstance(treeClass, children.size());
                Object m_Props = Array.newInstance(double.class, children.size());

                for (Node child : children) {
                    SimplePredicate predicate = (SimplePredicate) child.getPredicate();
                    int valueIndex = attribute.indexOfValue(predicate.getValue());
                    
                    Array.set(m_Successors, valueIndex, buildRandomTreeNode(tree, child));
                    Array.set(m_Props, valueIndex, getNodeTrainingProportion(child));
                }
                
                field = treeClass.getDeclaredField("m_Successors");
                field.setAccessible(true);
                field.set(treeNode, m_Successors);
                
                field = treeClass.getDeclaredField("m_Prop");
                field.setAccessible(true);
                field.set(treeNode, m_Props);
            } else {
                throw new RuntimeException("Attribute type not supported: " + attribute);
            }               
        }

        return treeNode;
    }
}
