# pmml-weka
PMML support for Weka classifiers, using Weka 3.7 and 3.9 implementation and JPMML framework. Currently only RandomForest classifier is supported.

For Weka 3.9, there is a .jar file in the release sections that, given a Weka .model file, returns a PMML file.

**Usage in Linux:**
java -jar weka-to-pmml-3.9.jar your_model.model
