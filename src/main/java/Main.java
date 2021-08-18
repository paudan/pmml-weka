import net.paudan.weka.pmml.PMMLConversionException;
import net.paudan.weka.pmml.RandomForestPMMLProducer;
import org.apache.commons.io.IOUtils;
import org.dmg.pmml.PMML;
import org.jpmml.model.JAXBUtil;
import weka.classifiers.trees.RandomForest;

import javax.xml.transform.stream.StreamResult;
import java.io.*;

public class Main {

    public static void main(String[] args)  //static method
    {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(args[0]));
            System.out.println("Loading Weka RandomForest model...");
            RandomForest rf = (RandomForest) ois.readObject();
            System.out.println("Weka RandomForest model Loaded.");
            ois.close();

            RandomForestPMMLProducer producer = new RandomForestPMMLProducer();

            System.out.println("Generating PMML Random Forest...");
            PMML pmml = producer.produce(rf);
            System.out.println("PMML Random Forest Generated.");

            String name = args[0].split("\\.")[0];

            OutputStream os = null;
            try {
                System.out.println("Exporting PMML Random Forest File...");
                os = new FileOutputStream(name + ".xml");
                StreamResult result = new StreamResult(os);
                JAXBUtil.marshalPMML(pmml, result);
                System.out.println("PMML Random Forest File exported.");
            } catch (Exception e) {
                System.err.println("Error: There was a problem generating the file " + name+ ".xml.");
            } finally {
                IOUtils.closeQuietly(os);
            }

        } catch (FileNotFoundException e) {
            System.err.println("Error: The file " + args[0] + "could not be found.");
        } catch (IOException e) {
            System.err.println("Error: There was an error while processing the file " + args[0] + ".");
        } catch (ClassNotFoundException | PMMLConversionException e) {
            e.printStackTrace();
        }
    }
}