package net.paudan.weka.pmml;

import org.dmg.pmml.PMML;
import weka.classifiers.Classifier;

import java.io.File;

public interface PMMLConsumer<T extends Classifier> {

    /**
     * Builds a new classifier from the given PMML String.
     * <p/>
     * The given {@code pmmlString} should be a valid PMML.
     *
     * @param pmmlString A String representing the PMML that is to be converted to a {@link hr.irb.fastRandomForest.FastRandomForest}.
     * @return A new {@link Classifier} instance.
     * @throws Exception If it fails to convert the given PMML to a {@link Classifier}.
     */
    T consume(String pmmlString) throws PMMLConversionException;

    /**
     * Builds a new classifier from the given file.
     *
     * @param file The file with the PMML representation of the classifier.
     * @return A new {@link Classifier} instance.
     * @throws Exception If it fails to convert the given file to a {@link Classifier}.
     */
    T consume(File file) throws PMMLConversionException;

    /**
     * Builds a new classifier from the given {@link org.dmg.pmml.PMML}.
     *
     * @param pmml The {@link org.dmg.pmml.PMML} which is to be converted to a {@link hr.irb.fastRandomForest.FastRandomForest}.
     * @return A new {@link Classifier} instance.
     * @throws Exception If it fails to convert the given PMML to a {@link Classifier}.
     */
    T consume(PMML pmml) throws PMMLConversionException;
}
