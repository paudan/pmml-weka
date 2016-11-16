package net.paudan.weka.pmml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.Arrays;
import static org.junit.Assert.assertEquals;
import org.junit.Test;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class BasicTest {
    
    @Test
    public void testIrisReflection() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        File dataset = new File(classLoader.getResource("credit-german.arff").getFile());
        //File dataset = Paths.get("test", "resources", "credit-german.arff").toFile();
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForest(), data);
            testConversion(classifier, data);
        }
    }
    
    @Test
    public void testShuttleReflection() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        File dataset = new File(classLoader.getResource("shuttle-landing-control.arff").getFile());
        //File dataset = Paths.get("test", "resources", "shuttle-landing-control.arff").toFile();
        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(dataset))) {
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            AbstractClassifier classifier = testAbstractClassifier(new RandomForest(), data);
            testConversion(classifier, data);
        }
    }
    
    protected AbstractClassifier testAbstractClassifier(AbstractClassifier classifier, Instances instances) throws Exception {
        classifier.setOptions(new String[]{"-I", "10", "-K", "0", "-S", "4", "-depth", "0", "-print"});
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
}
