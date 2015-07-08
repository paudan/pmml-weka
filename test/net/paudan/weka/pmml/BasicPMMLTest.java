package net.paudan.weka.pmml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Enumeration;
import static org.junit.Assert.assertEquals;
import org.junit.Test;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomForestWrapper;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class BasicPMMLTest {
    
    @Test
    public void testIrisReflection() throws Exception {
        File dataset = new File("test//resources//iris_model_builder.arff");
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForest(), data);
            testConversion(classifier, data);
        }
    }
    
    @Test
    public void testIrisWrapper() throws Exception {
        File dataset = new File("test//resources//iris_model_builder.arff");
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForestWrapper(), data);
            testConversion2(classifier, data);
        }
    }
    
    @Test
    public void testShuttleReflection() throws Exception {
        File dataset = new File("test//resources//shuttle-landing-control.arff");
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForest(), data);
            testConversion(classifier, data);
        }
    }
    
    @Test
    public void testShuttleWrapper() throws Exception {
        File dataset = new File("test//resources//shuttle-landing-control.arff");
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForestWrapper(), data);
            testConversion2(classifier, data);
        }
    }
    
    @Test
    public void testClassifierEquivalence() throws Exception {
        File dataset = new File("test//resources//iris_model_builder.arff");
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForest(), data);
            File file = Files.createTempFile("reflect_iris_pmml", ".xml").toFile();
            System.err.println("writing to file: " + file.getAbsolutePath());
            new RandomForestPMMLProducer().produce((RandomForest) classifier, file);
            RandomForest rf1 = new RandomForestPMMLConsumer().consume(file);
            
            AbstractClassifier classifier2 = testAbstractClassifier(new RandomForestWrapper(), data);
            file = Files.createTempFile("wrapper_frf_iris_pmml", ".xml").toFile();
            System.err.println("writing to file: " + file.getAbsolutePath());
            new net.paudan.weka.pmml.wrapper.RandomForestPMMLProducer().produce((RandomForestWrapper) classifier2, file);
            RandomForestWrapper rf2 = new net.paudan.weka.pmml.wrapper.RandomForestPMMLConsumer().consume(file);

            Enumeration<String> measures = rf1.enumerateMeasures();
            while (measures.hasMoreElements())
                System.out.println(measures.nextElement());
            System.out.println(rf2.toString());
        }
        
    }
    
    protected AbstractClassifier testAbstractClassifier(AbstractClassifier classifier, Instances instances) throws Exception {
        /*classifier.setOptions(new String[]{"-I", "1", "-K", "1", "-S", "1", "-depth", "1"});
        classifier.buildClassifier(instances);
        testConversion(classifier, instances);

        classifier.setOptions(new String[]{"-I", "2", "-K", "2", "-S", "2", "-depth", "2"});
        classifier.buildClassifier(instances);
        testConversion(classifier, instances);*/

        classifier.setOptions(new String[]{"-I", ""+instances.numInstances(), "-K", ""+instances.numInstances(), "-S", "4", "-depth", "0"});
        classifier.buildClassifier(instances);
        return classifier;
        
    }

    protected void testConversion(Classifier classifier, Instances instances) throws Exception {
        File file = Files.createTempFile("reflect_iris_pmml", ".xml").toFile();

        System.err.println("writing to file: " + file.getAbsolutePath());
        new RandomForestPMMLProducer().produce((RandomForest) classifier, file);
        Classifier fromPmml = new RandomForestPMMLConsumer().consume(file);
        for (int i = 0; i < instances.numInstances(); i++) {
            String wekaDist = Arrays.toString(classifier.distributionForInstance(instances.instance(i)));
            String pmmlDist = Arrays.toString(fromPmml.distributionForInstance(instances.instance(i)));

            assertEquals("Distributions for instance match.", wekaDist, pmmlDist);
        }

        //file.delete();
    }
    
    protected void testConversion2(Classifier classifier, Instances instances) throws Exception {
        File file = Files.createTempFile("wrapper_frf_iris_pmml", ".xml").toFile();

        System.err.println("writing to file: " + file.getAbsolutePath());
        new net.paudan.weka.pmml.wrapper.RandomForestPMMLProducer().produce((RandomForestWrapper) classifier, file);
        Classifier fromPmml = new net.paudan.weka.pmml.wrapper.RandomForestPMMLConsumer().consume(file);
        for (int i = 0; i < instances.numInstances(); i++) {
            String wekaDist = Arrays.toString(classifier.distributionForInstance(instances.instance(i)));
            String pmmlDist = Arrays.toString(fromPmml.distributionForInstance(instances.instance(i)));

            assertEquals("Distributions for instance match.", wekaDist, pmmlDist);
        }

        //file.delete();
    }
}
