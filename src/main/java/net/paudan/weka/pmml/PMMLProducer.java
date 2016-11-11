package net.paudan.weka.pmml;

import java.io.File;
import org.dmg.pmml.PMML;
import weka.classifiers.Classifier;

public interface PMMLProducer<T extends Classifier> {

    /**
     * Converts the given classifier instance to PMML and saves the result in the given {@link File}.
     *
     * @param classifier The {@link Classifier} instance to convert to PMML.
     * @param targetFile The file where to save the resulting PMML.
     * @throws PMMLConversionException If if fails to convert the classifier.
     */
    void produce(T classifier, File targetFile) throws PMMLConversionException;

    /**
     * Converts the given {@link Classifier} instance to PMML.
     *
     * @param classifier The {@link Classifier} instance to convert to PMML.
     * @return A {@link org.dmg.pmml.PMML} instance representing the PMML structure.
     */
    PMML produce(T classifier) throws PMMLConversionException;
}

