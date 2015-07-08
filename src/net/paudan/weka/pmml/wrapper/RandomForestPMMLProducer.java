package net.paudan.weka.pmml.wrapper;

import java.io.File;
import java.io.FileOutputStream;
import weka.core.Attribute;

import javax.xml.transform.stream.StreamResult;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import static net.paudan.weka.pmml.PMMLConversionCommons.TRAINING_PROPORTION_ELEMENT;
import static net.paudan.weka.pmml.PMMLConversionCommons.addScoreDistribution;
import static net.paudan.weka.pmml.PMMLConversionCommons.buildPMMLHeader;
import static net.paudan.weka.pmml.PMMLConversionCommons.leafScoreFromDistribution;
import net.paudan.weka.pmml.PMMLConversionException;
import net.paudan.weka.pmml.PMMLProducer;

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
import weka.classifiers.RandomForestWrapper;
import weka.classifiers.RandomTreeWrapper;
import weka.classifiers.Classifier;
import weka.classifiers.RandomForestUtils;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

/**
 * A producer that converts a {@link weka.classifiers.trees.RandomForest} instance to PMML.
 *
 * @author Ricardo Ferreira (ricardo.ferreira@feedzai.com), modifications by Paulius Danenas
 */
public class RandomForestPMMLProducer implements PMMLProducer<RandomForestWrapper> {

    private static final String ALGORITHM_NAME = "weka:"+RandomForest.class.getName();

    private static final String MODEL_NAME = ALGORITHM_NAME+"_Model";


    @Override
    public void produce(RandomForestWrapper randomForestClassifier, File targetFile) throws PMMLConversionException {
        PMML pmml = produce(randomForestClassifier);
        try (FileOutputStream fis = new FileOutputStream(targetFile)){
            JAXBUtil.marshalPMML(pmml, new StreamResult(fis));
        } catch (Exception e) {
            throw new PMMLConversionException("Failed to marshal the PMML to the given file.", e);
        }
    }

    @Override
    public PMML produce(RandomForestWrapper randomForestClassifier) {
        // Get the Instances from the first tree in the forest.
        Classifier[] baggingClassifiers = RandomForestUtils.getBaggingClassifiers(randomForestClassifier.getM_bagger());
        Instances data = ((RandomTreeWrapper) baggingClassifiers[0]).getM_Info();

        Header header = buildPMMLHeader("Weka RandomForest as PMML.");

        PMML pmml = new PMML("4.2", header, new DataDictionary());

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

        if (randomForestClassifier.getM_bagger() != null) {
            int segmentId = 1;
            for (Classifier classifier : RandomForestUtils.getBaggingClassifiers(randomForestClassifier.getM_bagger())) {
                Segment segment = buildSegment(miningSchema, segmentId++, (RandomTreeWrapper) classifier);
                segmentation.addSegments(segment);
            }
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
    private static Segment buildSegment(MiningSchema miningSchema, int segmentId, RandomTreeWrapper randomTree) {
        int rootNodeId = 1;

        Node rootNode = new Node();
        rootNode.setId(String.valueOf(rootNodeId));
        rootNode.setPredicate(new True());
        TreeModel treeModel = new TreeModel(MiningFunctionType.CLASSIFICATION, miningSchema, rootNode);
        treeModel.setAlgorithmName(ALGORITHM_NAME);
        treeModel.setModelName(MODEL_NAME);
        treeModel.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

        buildTreeNode(randomTree, randomTree.getM_Tree(), rootNodeId, rootNode);

        Segment segment = new Segment();
        segment.setId(String.valueOf(segmentId));
        segment.setModel(treeModel);

        return segment;
    }

    /**
     * Builds a new {@link org.dmg.pmml.Node PMML Node} from the given {@link RandomTree.Tree Weka Tree Node}.
     *
     * @param tree           The {@link weka.classifiers.trees.RandomTree Weka RandomTree} being converted to a {@link org.dmg.pmml.PMML TreeModel}.
     * @param node           The Id to give to the generted {@link org.dmg.pmml.Node PMML Node}.
     * @param nodeId         The Id to give to the generted {@link org.dmg.pmml.Node PMML Node}.
     * @param parentPMMLNode The parent {@link org.dmg.pmml.Node PMML Node}.
     * @return The incremented Id given to recursively created {@link org.dmg.pmml.Node PMML Nodes}.
     */
    private static int buildTreeNode(RandomTreeWrapper tree, RandomTreeWrapper.TreeWrapper node, int nodeId, Node parentPMMLNode) {
        addScoreDistribution(parentPMMLNode, node.getM_ClassDistribution(), tree.getM_Info());

        if (node.getM_Attribute() == -1) {
            // Leaf: Add the node's score.
            parentPMMLNode.setScore(leafScoreFromDistribution(node.getM_ClassDistribution(), tree.getM_Info()));
            return nodeId;
        }

        Attribute attribute = tree.getM_Info().attribute(node.getM_Attribute());

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
    private static int buildNominalNode(RandomTreeWrapper tree, Attribute attribute, 
            RandomTreeWrapper.TreeWrapper node, int nodeId, Node parentPMMLNode) {
        List<Object> values = new ArrayList<>();
        Enumeration enumeration = attribute.enumerateValues();
        while (enumeration.hasMoreElements()) {
            values.add(enumeration.nextElement());
        }

        List<Node> children = new ArrayList<>();

        for (int i = 0; i < values.size(); i++) {
            Object value = values.get(i);

            SimplePredicate predicate = new SimplePredicate(new FieldName(attribute.name()), SimplePredicate.Operator.EQUAL);
            predicate.setValue(String.valueOf(value));
            Node child = new Node();
            child.setId(String.valueOf(++nodeId));
            child.setPredicate(predicate);

            nodeId = buildTreeNode(tree, node.getM_Successors()[i], nodeId, child);

            // Training proportion extension.
            Extension ext = new Extension();
            ext.setName(TRAINING_PROPORTION_ELEMENT);
            ext.setValue(String.valueOf(node.getM_Prop()[i]));
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
     * @param nodeId         The Id to give to the generted {@link org.dmg.pmml.Node PMML Node}.
     * @param parentPMMLNode The parent {@link org.dmg.pmml.Node PMML Node}.
     * @return The incremented Id given to recursively created {@link org.dmg.pmml.Node PMML Nodes}.
     */
    private static int buildNumericNode(RandomTreeWrapper tree, Attribute attribute, 
            RandomTreeWrapper.TreeWrapper node, int nodeId, Node parentPMMLNode) {
        SimplePredicate predicateLo = new SimplePredicate(new FieldName(attribute.name()), 
                SimplePredicate.Operator.LESS_THAN);
        predicateLo.setValue(String.valueOf(node.getM_SplitPoint()));
        SimplePredicate predicateHi = new SimplePredicate(new FieldName(attribute.name()), 
                SimplePredicate.Operator.GREATER_OR_EQUAL);
        predicateHi.setValue(String.valueOf(node.getM_SplitPoint()));

        Node nodeLo = new Node();
        nodeLo.setId(String.valueOf(++nodeId));
        nodeLo.setPredicate(predicateLo);

        nodeId = buildTreeNode(tree, node.getM_Successors()[0], nodeId, nodeLo);

        Node nodeHi = new Node();
        nodeHi.setId(String.valueOf(++nodeId));
        nodeHi.setPredicate(predicateHi);

        nodeId = buildTreeNode(tree, node.getM_Successors()[1], nodeId, nodeHi);

        // Training proportion extension.
        Extension ext1 = new Extension();
        ext1.setName(TRAINING_PROPORTION_ELEMENT);
        ext1.setValue(String.valueOf(node.getM_Prop()[0]));
        Extension ext2 = new Extension();
        ext2.setName(TRAINING_PROPORTION_ELEMENT);
        ext2.setValue(String.valueOf(node.getM_Prop()[1]));
        nodeLo.addExtensions(ext1, ext2);

        parentPMMLNode.addNodes(nodeLo, nodeHi);

        return nodeId;
    }
}

