package net.paudan.weka.pmml;

public class WekaClassifierException extends Exception {

    public WekaClassifierException(Throwable e) {
        super(e);
    }

    public WekaClassifierException(String msg) {
        super(msg);
    }


    public WekaClassifierException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
