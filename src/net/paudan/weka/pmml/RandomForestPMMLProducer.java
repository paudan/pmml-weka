package net.paudan.weka.pmml;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
import weka.core.Attribute;
import javax.xml.transform.stream.StreamResult;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import static net.paudan.weka.pmml.PMMLConversionCommons.TRAINING_PROPORTION_ELEMENT;
import static net.paudan.weka.pmml.PMMLConversionCommons.addScoreDistribution;
import static net.paudan.weka.pmml.PMMLConversionCommons.buildPMMLHeader;
import static net.paudan.weka.pmml.PMMLConversionCommons.leafScoreFromDistribution;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Extension;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.Header;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Node;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Segment;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.TreeModel;
import org.dmg.pmml.True;
import org.dmg.pmml.Value;
import org.jpmml.model.JAXBUtil;
import weka.classifiers.Classifier;
import weka.classifiers.RandomForestUtils;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

/**
 * A producer that converts a {@link weka.classifiers.trees.RandomForest} instance to PMML.
 *
 * @author Paulius Danenas (danpaulius@gmail.com), based on code by Ricardo Ferreira (ricardo.ferreira@feedzai.com)
 */
public class RandomForestPMMLProducer implements PMMLProducer<RandomForest> {

    private static final String ALGORITHM_NAME = "weka:"+RandomForest.class.getName();

    private static final String MODEL_NAME = ALGORITHM_NAME+"_Model";


    @Override
    public void produce(RandomForest randomForestClassifier, File targetFile) throws PMMLConversionException {
        PMML pmml = produce(randomForestClassifier);
        try (FileOutputStream fis = new FileOutputStream(targetFile)){
            JAXBUtil.marshalPMML(pmml, new StreamResult(fis));
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to marshal the PMML to the given file.", e);
        }
    }

    @Override
    public PMML produce(RandomForest randomForestClassifier) throws PMMLConversionException  {
        Header header = buildPMMLHeader("Weka RandomForest as PMML");
        PMML pmml = new PMML("4.2", header, new DataDictionary());
        try {
            // Get the Instances from the first tree in the forest.
            Field field = RandomForest.class.getDeclaredField("m_bagger");
            field.setAccessible(true);
            Bagging bagger = (Bagging) field.get(randomForestClassifier);
            
            Classifier[] baggingClassifiers = RandomForestUtils.getBaggingClassifiers(bagger);
            field = RandomTree.class.getDeclaredField("m_Info");
            field.setAccessible(true);
            Instances data = (Instances) field.get((RandomTree) baggingClassifiers[0]);

            // Builds the PMML DataDictionary and MiningSchema elements.
            DataDictionary dataDictionary = new DataDictionary();
            MiningSchema miningSchema = new MiningSchema();
            
            if (data != null) {
                for (int i = 0; i < data.numAttributes(); i++) {
                    Attribute attribute = data.attribute(i);
                    
                    DataType fieldType;
                    if (attribute.isNumeric()) {
                        fieldType = DataType.DOUBLE;
                    } else {
                        fieldType = DataType.STRING;
                    }
                    
                    DataField dataField = new DataField(new FieldName(attribute.name()), attribute.isNominal() ?
                            OpType.CATEGORICAL : OpType.CONTINUOUS, fieldType);
                    if (attribute.isNominal()) {
                        Enumeration enumeration = attribute.enumerateValues();
                        while (enumeration.hasMoreElements()) {
                            dataField.addValues(new Value(String.valueOf(enumeration.nextElement())));
                        }
                    }
                    
                    dataDictionary.addDataFields(dataField);
                    
                    MiningField miningField = new MiningField(new FieldName(attribute.name()));
                    
                    if (data.classIndex() == i) {
                        miningField.setUsageType(FieldUsageType.PREDICTED);
                    } else {
                        miningField.setUsageType(FieldUsageType.ACTIVE);
                    }
                    miningSchema.addMiningFields(miningField);
                }
            }
            
            pmml.setDataDictionary(dataDictionary);
            
            MiningModel miningModel = new MiningModel(MiningFunctionType.CLASSIFICATION, miningSchema);
            miningModel.setModelName(MODEL_NAME);
            
            pmml.addModels(miningModel);
            
            Segmentation segmentation = new Segmentation();
            segmentation.setMultipleModelMethod(MultipleModelMethodType.MAJORITY_VOTE);
            miningModel.setSegmentation(segmentation);
            
            if (bagger != null) {
                int segmentId = 1;
                for (Classifier classifier : RandomForestUtils.getBaggingClassifiers(bagger)) {
                    Segment segment = buildSegment(miningSchema, segmentId++, (RandomTree) classifier);
                    segmentation.addSegments(segment);
                }
            }
        } catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException | ClassNotFoundException ex) {
            throw new PMMLConversionException(ex);
        } 
        return pmml;
    }


    /**
     * Builds a {@link org.dmg.pmml.Segment PMML Segment} that contains the {@link org.dmg.pmml.TreeModel PMML TreeModel}
     * representing the given {@link weka.classifiers.trees.RandomTree Weka RandomTree}.
     *
     * @param miningSchema The {@link org.dmg.pmml.MiningSchema PMML MiningSchema} that lists fields as used in the model.
     * @param segmentId    The Id to given to the {@link org.dmg.pmml.Segment PMML Segment element}.
     * @param randomTree   The {@link weka.classifiers.trees.RandomTree Weka RandomTree} to be converted to a {@link org.dmg.pmml.TreeModel PMML TreeModel}.
     * @return The created {@link org.dmg.pmml.Segment PMML Segment}.
     */
    private static Segment buildSegment(MiningSchema miningSchema, int segmentId, RandomTree randomTree) 
            throws IllegalArgumentException, IllegalAccessException, NoSuchFieldException, ClassNotFoundException {
        int rootNodeId = 1;

        Node rootNode = new Node();
        rootNode.setId(String.valueOf(rootNodeId));
        rootNode.setPredicate(new True());
        TreeModel treeModel = new TreeModel(MiningFunctionType.CLASSIFICATION, miningSchema, rootNode);
        treeModel.setAlgorithmName(ALGORITHM_NAME);
        treeModel.setModelName(MODEL_NAME);
        treeModel.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

        Field field = RandomTree.class.getDeclaredField("m_Tree");
        field.setAccessible(true);
        Object bagger = field.get(randomTree);
        buildTreeNode(randomTree, bagger, rootNodeId, rootNode);

        Segment segment = new Segment();
        segment.setId(String.valueOf(segmentId));
        segment.setModel(treeModel);

        return segment;
    }

    /**
     * Builds a new {@link org.dmg.pmml.Node PMML Node} from the given {@link RandomTree.Tree Weka Tree Node}.
     *
     * @param tree           The {@link weka.classifiers.trees.RandomTree Weka RandomTree} being converted to a {@link org.dmg.pmml.PMML TreeModel}.
     * @param node           The Id to give to the generated {@link org.dmg.pmml.Node PMML Node}.
     * @param nodeId         The Id to give to the generated {@link org.dmg.pmml.Node PMML Node}.
     * @param parentPMMLNode The parent {@link org.dmg.pmml.Node PMML Node}.
     * @return The incremented Id given to recursively created {@link org.dmg.pmml.Node PMML Nodes}.
     */
    private static int buildTreeNode(RandomTree tree, Object node, int nodeId, Node parentPMMLNode) 
            throws NoSuchFieldException, IllegalArgumentException, IllegalAccessException, ClassNotFoundException {
        Field field = RandomTree.class.getDeclaredField("m_Info");
        field.setAccessible(true);
        Instances m_info = (Instances) field.get(tree);
        
        Class<?> treeClass = Class.forName("weka.classifiers.trees.RandomTree$Tree");
        field = treeClass.getDeclaredField("m_ClassDistribution");
        field.setAccessible(true);
        double[] m_classDistribution = double[].class.cast(field.get(node));
        field = treeClass.getDeclaredField("m_Attribute");
        field.setAccessible(true);
        int m_attribute = field.getInt(node);
        
        addScoreDistribution(parentPMMLNode, m_classDistribution, m_info);

        if (m_attribute == -1) {
            // Leaf: Add the node's score.
            parentPMMLNode.setScore(leafScoreFromDistribution(m_classDistribution, m_info));
            return nodeId;
        }

        Attribute attribute = m_info.attribute(m_attribute);

        if (attribute.isNominal()) {
            return buildNominalNode(tree, attribute, node, nodeId, parentPMMLNode);
        } else if (attribute.isNumeric()) {
            return buildNumericNode(tree, attribute, node, nodeId, parentPMMLNode);
        } else {
            throw new RuntimeException("Unsupported attribute type for: " + attribute);
        }
    }

    /**
     * Builds the {@link org.dmg.pmml.Node PMML Node} for a nominal attribute.
     * <p/>
     * In PMML these nodes are represented with multiple children, one for each of the attribute's values.
     * <p/>
     * For example, consider a nominal attribute, named "nominalAttribute", with values "cold", "hot" and "warm". In PMML this translates to:
     * <pre>
     *     {@code
     *       <Node id="2" score="1">
     *         <SimplePredicate field="nominalAttribute" operator="equal" value="cold"/>
     *       </Node>
     *       <Node id="3" score="0">
     *         <SimplePredicate field="nominalAttribute" operator="equal" value="hot"/>
     *       </Node>
     *       <Node id="4" score="1.5">
     *         <SimplePredicate field="nominalAttribute" operator="equal" value="warm"/>
     *       </Node>
     *     }
     * </pre>
     *
     * @param tree           The {@link weka.classifiers.trees.RandomTree Weka RandomTree} being converted to a {@link org.dmg.pmml.PMML TreeModel}.
     * @param attribute      The {@link weka.core.Attribute} to which the node to build refers to.
     * @param node           The {@link weka.classifiers.trees.RandomTree.Tree Weka RandomTree node} we are converting to PMML.
     * @param nodeId         The Id to give to the generted {@link org.dmg.pmml.Node PMML Node}.
     * @param parentPMMLNode The parent {@link org.dmg.pmml.Node PMML Node}.
     * @return The incremented Id given to recursively created {@link org.dmg.pmml.Node PMML Nodes}.
     */
    private static int buildNominalNode(RandomTree tree, Attribute attribute, Object node, int nodeId, Node parentPMMLNode) 
            throws ClassNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
        List<Object> values = new ArrayList<>();
        Enumeration enumeration = attribute.enumerateValues();
        while (enumeration.hasMoreElements()) {
            values.add(enumeration.nextElement());
        }
        
        Class<?> treeClass = Class.forName("weka.classifiers.trees.RandomTree$Tree");
        Field field = treeClass.getDeclaredField("m_Successors");
        field.setAccessible(true);
        Object[] m_Successors = Object[].class.cast(field.get(node));
        field = treeClass.getDeclaredField("m_Prop");
        field.setAccessible(true);
        double[] m_Prop = double[].class.cast(field.get(node));

        List<Node> children = new ArrayList<>();

        for (int i = 0; i < values.size(); i++) {
            Object value = values.get(i);

            SimplePredicate predicate = new SimplePredicate(new FieldName(attribute.name()), SimplePredicate.Operator.EQUAL);
            predicate.setValue(String.valueOf(value));
            Node child = new Node();
            child.setId(String.valueOf(++nodeId));
            child.setPredicate(predicate);

            nodeId = buildTreeNode(tree, m_Successors[i], nodeId, child);

            // Training proportion extension.
            Extension ext = new Extension();
            ext.setName(TRAINING_PROPORTION_ELEMENT);
            ext.setValue(String.valueOf(m_Prop[i]));
            child.addExtensions(ext);

            children.add(child);
        }

        parentPMMLNode.addNodes(children.toArray(new Node[] {}));

        return nodeId;
    }

    /**
     * Builds the {@link org.dmg.pmml.Node PMML Node} for a numeric attribute.
     * <p/>
     * In PMML these nodes are represented having two children, each with a predicate that checks the node's split point.
     * <p/>
     * For example, consider a numeric attribute, named "numericAttribute", with a split point of 2.5 and two leaf nodes. In PMML this translates to:
     * <pre>
     *     {@code
     *       <Node id="2" score="1">
     *         <SimplePredicate field="numericAttribute" operator="lessThan" value="2.5"/>
     *       </Node>
     *       <Node id="3" score="0">
     *         <SimplePredicate field="numericAttribute" operator="greaterOrEqual" value="2.5"/>
     *       </Node>
     *     }
     * </pre>
     *
     * @param tree           The {@link weka.classifiers.trees.RandomTree Weka RandomTree} being converted to a {@link org.dmg.pmml.PMML TreeModel}.
     * @param attribute      The {@link weka.core.Attribute} to which the node to build refers to.
     * @param node           The {@link weka.classifiers.trees.RandomTree.Tree Weka RandomTree node} we are converting to PMML.
     * @param nodeId         The Id to give to the generated {@link org.dmg.pmml.Node PMML Node}.
     * @param parentPMMLNode The parent {@link org.dmg.pmml.Node PMML Node}.
     * @return The incremented Id given to recursively created {@link org.dmg.pmml.Node PMML Nodes}.
     */
    private static int buildNumericNode(RandomTree tree, Attribute attribute, Object node, int nodeId, Node parentPMMLNode) 
            throws ClassNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
        Class<?> treeClass= Class.forName("weka.classifiers.trees.RandomTree$Tree");
        Field field = treeClass.getDeclaredField("m_Successors");
        field.setAccessible(true);
        Object[] m_Successors = Object[].class.cast(field.get(node));
        field = treeClass.getDeclaredField("m_Prop");
        field.setAccessible(true);
        double[] m_Prop = double[].class.cast(field.get(node));
        field = treeClass.getDeclaredField("m_SplitPoint");
        field.setAccessible(true);
        double m_SplitPoint = field.getDouble(node);
        
        SimplePredicate predicateLo = new SimplePredicate(new FieldName(attribute.name()), 
                SimplePredicate.Operator.LESS_THAN);
        predicateLo.setValue(String.valueOf(m_SplitPoint));
        SimplePredicate predicateHi = new SimplePredicate(new FieldName(attribute.name()), 
                SimplePredicate.Operator.GREATER_OR_EQUAL);
        predicateHi.setValue(String.valueOf(m_SplitPoint));

        Node nodeLo = new Node();
        nodeLo.setId(String.valueOf(++nodeId));
        nodeLo.setPredicate(predicateLo);

        nodeId = buildTreeNode(tree, m_Successors[0], nodeId, nodeLo);

        Node nodeHi = new Node();
        nodeHi.setId(String.valueOf(++nodeId));
        nodeHi.setPredicate(predicateHi);

        nodeId = buildTreeNode(tree, m_Successors[1], nodeId, nodeHi);

        // Training proportion extension.
        Extension ext1 = new Extension();
        ext1.setName(TRAINING_PROPORTION_ELEMENT);
        ext1.setValue(String.valueOf(m_Prop[0]));
        Extension ext2 = new Extension();
        ext2.setName(TRAINING_PROPORTION_ELEMENT);
        ext2.setValue(String.valueOf(m_Prop[1]));
        nodeLo.addExtensions(ext1, ext2);

        parentPMMLNode.addNodes(nodeLo, nodeHi);

        return nodeId;
    }
}

