package net.paudan.weka.pmml;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

import java.util.List;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Extension;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.Header;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.Node;
import static org.dmg.pmml.OpType.CATEGORICAL;
import static org.dmg.pmml.OpType.CONTINUOUS;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.TreeModel;
import org.dmg.pmml.Value;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Common methods used by Weka's PMML consumer and producers.
 *
 * @author Ricardo Ferreira (ricardo.ferreira@feedzai.com)
 */
public final class PMMLUtils {

    public enum Algorithm {

        RANDOM_FOREST {

            @Override
            public PMMLProducer getPMMLProducer() {
                return new RandomForestPMMLProducer();
            }

            @Override
            public Class<? extends Classifier> getClassifierClass() {
                return RandomForest.class;
            }

            @Override
            public PMMLConsumer getPMMLConsumer() {
                return null;
            }
        };

        /**
         * Retrieves a {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLConsumer} instance
         * used to convert PMML into classifiers of this type.
         *
         * @return A {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLConsumer} instance.
         */
        public abstract PMMLConsumer getPMMLConsumer();

        /**
         * Retrieves a {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLProducer} instance
         * used to convert classifiers of this type into PMML.
         *
         * @return A {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLProducer} instance.
         */
        public abstract PMMLProducer getPMMLProducer();

        /**
         * Retrieves the {@link weka.classifiers.Classifier} class represented by this type.
         *
         * @return The class of the classifier represented by this type.
         */
        public abstract Class<? extends Classifier> getClassifierClass();

        /**
         * Retrieves the {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLConversionCommons.Algorithm} associated
         * with the given classifier.
         *
         * @param classifier The classifier for which to retrieve the associated {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLConversionCommons.Algorithm}.
         * @return The {@link Algorithm}.
         * @throws PMMLConversionException If the classifier has no corresponding {@link com.feedzai.fos.impl.weka.utils.pmml.PMMLConversionCommons.Algorithm}.
         */
        public static Algorithm fromClassifier(Classifier classifier) throws PMMLConversionException {
            Class klass = classifier.getClass();

            for (Algorithm algorithm : Algorithm.values()) {
                if (klass.equals(algorithm.getClassifierClass())) {
                    return algorithm;
                }
            }

            throw new PMMLConversionException("Unsupported classifier '" + classifier.getClass().getSimpleName() + "'.");
        }
    }

    /**
     * The name of the PMML element with information about the algorithm the model was trained with.
     */
    public static final String ALGORITHM_EXTENSION_ELEMENT = "algorithm";

    /**
     * The name of the PMML element with information about the {@link weka.classifiers.trees.RandomTree.Tree#m_Prop}.
     */
    public static final String TRAINING_PROPORTION_ELEMENT = "trainingProportion";


    /**
     * Creates a new {@link org.dmg.pmml.Header} element.
     *
     * @param description The description to have in the header.
     * @return A new {@link org.dmg.pmml.Header} element.
     */
    public static Header buildPMMLHeader(String description) {
        Header header = new Header();
        header.setCopyright("www.dmg.org");
        header.setDescription(description);
        return header;     
    }

    /**
     * Adds a {@link org.dmg.pmml.ScoreDistribution PMML ScoreDistribution element} to the given {@link org.dmg.pmml.Node PMML Node}
     * with the confidence from the given {@code classDistribution}.
     *
     * @param pmmlNode          The {@link org.dmg.pmml.Node PMML Node} to which add a {@link org.dmg.pmml.ScoreDistribution PMML ScoreDistribution element}.
     * @param classDistribution The class distribution to calculate a score for.
     * @param instances         The header {@link weka.core.Instances}.
     */
    public static void addScoreDistribution(Node pmmlNode, double[] classDistribution, Instances instances) {
        if (classDistribution != null) {

            double sum = 0.0;
            for (int i = 0; i < classDistribution.length; i++) {
                sum += classDistribution[i];
            }

            for (int i = 0; i < classDistribution.length; i++) {
                String value = instances.classAttribute().value(i);
                double probability;

                if (sum != 0.0) {
                    probability = classDistribution[i]/sum;
                } else {
                    probability = 1.0/classDistribution.length;
                }

                ScoreDistribution scoreDistribution = new ScoreDistribution(value, 0);
                scoreDistribution.setConfidence(classDistribution[i]);
                scoreDistribution.setProbability(probability);
                pmmlNode.addScoreDistributions(scoreDistribution);
            }
        }
    }

    /**
     * Retrieves a String representing the score of the given class distribution.
     *
     * @param classDistribution The class distribution to calculate a score for.
     * @param instances         The header {@link weka.core.Instances}.
     * @return A String representing the score of the given class distribution.
     */
    public static String leafScoreFromDistribution(double[] classDistribution, Instances instances) {
        double sum = 0, maxCount = 0;
        int maxIndex = 0;
        if (classDistribution != null) {
            sum = Utils.sum(classDistribution);
            maxIndex = Utils.maxIndex(classDistribution);
            maxCount = classDistribution[maxIndex];
        }
        return instances.classAttribute().value(maxIndex);
    }

    /**
     * Retrieves the class distribution for the given node.
     * <p/>
     * We represent this in PMML as the confidence of a list of {@link org.dmg.pmml.ScoreDistribution} elements.
     *
     * @param node The {@link org.dmg.pmml.Node PMML node} for which to retrieve the class distribution.
     * @return A array with the class distribution for the given node.
     */
    public static double[] getClassDistribution(Node node) {
        List<ScoreDistribution> scoreDistributions = node.getScoreDistributions();

        double[] classDistribution = null;

        if (!scoreDistributions.isEmpty()) {
            classDistribution = new double[scoreDistributions.size()];

            for (int i = 0; i < scoreDistributions.size(); i++) {
                classDistribution[i] = scoreDistributions.get(i).getConfidence();
            }
        }

        return classDistribution;
    }

    /**
     * Retrieves the training proportion of the given node.
     * <p/>
     * We represent this value in PMML as an {@link org.dmg.pmml.Extension} element of the node.
     *
     * @param node The {@link org.dmg.pmml.Node PMML node} for which to retrieve the training proportion.
     * @return The training proportion of the given node.
     */
    public static double getNodeTrainingProportion(Node node) {
        for (Extension extension : node.getExtensions()) {
            if (TRAINING_PROPORTION_ELEMENT.equals(extension.getName())) {
                return Double.valueOf(extension.getValue());
            }
        }

        return 0;
    }

    /**
     * Retrieves the index of the class attribute. This is the attribute to be predicted.
     *
     * @param instances The {@link weka.core.Instances}.
     * @param treeModel The {@link org.dmg.pmml.TreeModel PMML TreeModel element}.
     * @return An int representing the index of the class attribute.
     */
    public static int getClassIndex(Instances instances, TreeModel treeModel) {
        MiningSchema miningModel = treeModel.getMiningSchema();

        String className = null;
        for (MiningField miningField : miningModel.getMiningFields()) {
            if (miningField.getUsageType() == FieldUsageType.PREDICTED) {
                className = miningField.getName().getValue();
                break;
            }
        }

        return instances.attribute(className).index();
    }

    /**
     * Retrieves the index of the class attribute. This is the attribute to be predicted.
     *
     * @param instances    The {@link weka.core.Instances}.
     * @param miningSchema The {@link org.dmg.pmml.MiningSchema PMML MiningSchema element}.
     * @return An int representing the index of the class attribute.
     */
    public static int getClassIndex(Instances instances, MiningSchema miningSchema) {
        String className = null;
        for (MiningField miningField : miningSchema.getMiningFields()) {
            if (miningField.getUsageType() == FieldUsageType.PREDICTED) {
                className = miningField.getName().getValue();
                break;
            }
        }

        return instances.attribute(className).index();
    }

    /**
     * Creates a new {@link weka.core.Instances} instance from the attributes in a {@link org.dmg.pmml.DataDictionary PMML DataDictionary}.
     *
     * @param dataDict The {@link org.dmg.pmml.DataDictionary PMML DataDictionary} from which to build instances.
     * @return A new {@link weka.core.Instances} instance.
     */
    public static Instances buildInstances(DataDictionary dataDict) {
        List<weka.core.Attribute> attributes = buildAttributes(dataDict);

        FastVector fastVector = new FastVector(attributes.size());
        for (weka.core.Attribute attribute : attributes) {
            fastVector.addElement(attribute);
        }

        return new Instances("instances", fastVector, attributes.size());
    }

    /**
     * Creates and returns {@link weka.core.Attribute attributes} from a {@link org.dmg.pmml.DataDictionary PMML DataDictionary}.
     *
     * @param dataDict The {@link org.dmg.pmml.DataDictionary PMML DataDictionary} from which to build the attributes.
     * @return A new immutable list of {@link weka.core.Attribute attributes}.
     */
    public static List<weka.core.Attribute> buildAttributes(DataDictionary dataDict) {
        List<weka.core.Attribute> attribuesBuilder = new ArrayList<>();

        for (DataField dataField : dataDict.getDataFields()) {
            attribuesBuilder.add(buildAttribute(dataField));
        }

        return attribuesBuilder;
    }

    /**
     * Creates a new {@link weka.core.Attribute} from a {@link org.dmg.pmml.DataField PMML DataField}.
     *
     * @param dataField The {@link org.dmg.pmml.DataField PMML DataField} representing the attribute.
     * @return A new {@link weka.core.Attribute}.
     */
    public static weka.core.Attribute buildAttribute(DataField dataField) {
        switch (dataField.getOpType()) {
            case CONTINUOUS:
                return new weka.core.Attribute(dataField.getName().getValue());
            case CATEGORICAL:
                List<Value> values = dataField.getValues();
                FastVector nominalValues = new FastVector();
                for (Value value : values) {
                    nominalValues.addElement(value.getValue());
                }
                return new weka.core.Attribute(dataField.getName().getValue(), nominalValues);
            default:
                throw new RuntimeException("PMML DataField OPTYPE " + dataField.getOpType() + " not supported.");
        }
    }

    /**
     * Retrieves the {@link org.dmg.pmml.MiningModel PMML MiningModel} element in a {@link org.dmg.pmml.PMML}.
     *
     * @param pmml The {@link org.dmg.pmml.PMML} from which to retrieve the {@link org.dmg.pmml.MiningModel PMML MiningModel}.
     * @return A {@link org.dmg.pmml.MiningModel PMML MiningModel}.
     */
    public static MiningModel getMiningModel(PMML pmml) {
        for (Model model : pmml.getModels()) {
            if (model instanceof MiningModel) {
                return (MiningModel) model;
            }
        }
        throw new RuntimeException("PMML MiningModel not found.");
    }
}
