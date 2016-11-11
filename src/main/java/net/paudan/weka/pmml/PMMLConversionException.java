package net.paudan.weka.pmml;

public class PMMLConversionException extends Exception {

    public PMMLConversionException(String message) {
        super(message);
    }

    public PMMLConversionException(String message, Throwable t) {
        super(message, t);
    }

    public PMMLConversionException(Throwable e) {
        super(e);
    }
}

