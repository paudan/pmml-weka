import net.paudan.weka.pmml.RandomForestPMMLConsumer;
import net.paudan.weka.pmml.RandomForestPMMLProducer;
import org.apache.commons.io.IOUtils;
import org.dmg.pmml.PMML;
import org.jpmml.model.JAXBUtil;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;

import javax.xml.transform.stream.StreamResult;
import java.io.*;

public class Main {

    public static void main(String[] args)  //static method
    {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("taiga2.model"));
            RandomForest rf = (RandomForest) ois.readObject();
            ois.close();

            RandomForestPMMLProducer producer = new RandomForestPMMLProducer();

            PMML pmml = producer.produce(rf);

            OutputStream os = null;
            try {
                os = new FileOutputStream("model.pmml");
                StreamResult result = new StreamResult(os);
                JAXBUtil.marshalPMML(pmml, result);
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                IOUtils.closeQuietly(os);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}