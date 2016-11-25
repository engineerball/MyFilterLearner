# MyFilterLearner

Example java application for learn a weka classifier step with test and train data


##### How to compile

prefer a weka has intalled at /opt/weka and weka.jar at /opt/weka/weka.jar

```
javac -cp /opt/weka/weka.jar MyFilteredLearner.java
```

##### How to run
```
java  -cp /opt/weka/weka.jar:. MyFilteredLearner imdbtrain.arff test.model
```
