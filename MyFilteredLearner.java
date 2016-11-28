/**
 * A Java class that implements a simple text learner, based on WEKA.
 * To be used with MyFilteredClassifier.java.
 * WEKA is available at: http://www.cs.waikato.ac.nz/ml/weka/
 * Copyright (C) 2013 Jose Maria Gomez Hidalgo - http://www.esp.uem.es/jmgomez
 *
 * This program is free software: you can redistribute it and/or modify
 * it for any purpose.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.*;


/**
 * This class implements a simple text learner in Java using WEKA.
 * It loads a text dataset written in ARFF format, evaluates a classifier on it,
 * and saves the learnt model for further use.
 * @author Jose Maria Gomez Hidalgo - http://www.esp.uem.es/jmgomez
 * @see MyFilteredClassifier
 */
public class MyFilteredLearner {

	/**
	 * Object that stores training data.
	 */
	Instances trainData;
	/**
	 * Object that stores test data.
	 */
	Instances testData;
	/**
	 * Object that stores the filter
	 */
	StringToWordVector filter;
	/**
	 * Object that stores the classifier
	 */
	FilteredClassifier classifier;

	/**
	 * This method loads a dataset in ARFF format. If the file does not exist, or
	 * it has a wrong format, the attribute trainData is null.
	 * @param fileName The name of the file that stores the dataset.
	 */
	public void loadDataset(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			trainData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		}
		catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
	}

	/**
	 * This method loads a dataset in ARFF format. If the file does not exist, or
	 * it has a wrong format, the attribute trainData is null.
	 * @param fileName The name of the file that stores the dataset.
	 */
	public void loadTestDataset(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			testData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		}
		catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
	}

	
	public void loadCSVData(String fileName) {
		String line = "";
        String cvsSplitBy = ",";

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {

            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] data = line.split(cvsSplitBy);

                System.out.println("Data [code= " + data[0] + " , name=" + data[1] + " , BC=" + data[2] + " , CC=" + data[3] + " , DC=" + data[4] + "]");

            }

        } catch (IOException e) {
            e.printStackTrace();
        }
		
	}
	
	/**
	 * This method evaluates the classifier. As recommended by WEKA documentation,
	 * the classifier is defined but not trained yet. Evaluation of previously
	 * trained classifiers can lead to unexpected results.
	 */
	public void evaluate() {
		try {
			trainData.setClassIndex(trainData.numAttributes() - 1);
			testData.setClassIndex(testData.numAttributes() - 1);
			// trainData.setClassIndex(trainData.numAttributes() - 1);
			filter = new StringToWordVector();
			filter.setAttributeIndices("first");
			classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			classifier.setClassifier(new RandomForest());
			classifier.buildClassifier(trainData);
			// System.out.println(testData);
			// String options[] = {"-v"};
			Evaluation eval = new Evaluation(trainData);
			// eval.crossValidateModel(classifier, trainData, 4, new Random(1));
			eval.evaluateModel(classifier, testData);
			
			int numTestInstances = testData.numInstances();
			System.out.printf("There are %d test instances\n", numTestInstances);
			// Loop over each test instance.
		    for (int i = 0; i < numTestInstances; i++)
		    {
		        // Get the true class label from the instance's own classIndex.
		        String trueClassLabel = 
		            testData.instance(i).toString(testData.classIndex());

		        // Make the prediction here.
		        double predictionIndex = 
		            classifier.classifyInstance(testData.instance(i)); 

		        // Get the predicted class label from the predictionIndex.
		        String predictedClassLabel =
		        		testData.classAttribute().value((int) predictionIndex);

		        // Get the prediction probability distribution.
		        double[] predictionDistribution = 
		            classifier.distributionForInstance(testData.instance(i)); 

		        // Print out the true label, predicted label, and the distribution.
		        System.out.printf("%5d: true=%-10s, predicted=%-10s, distribution=", 
		                          i, trueClassLabel, predictedClassLabel); 

		        // Loop over all the prediction labels in the distribution.
		        for (int predictionDistributionIndex = 0; 
		             predictionDistributionIndex < predictionDistribution.length; 
		             predictionDistributionIndex++)
		        {
		            // Get this distribution index's class label.
		            String predictionDistributionIndexAsClassLabel = 
		            		testData.classAttribute().value(
		                    predictionDistributionIndex);

		            // Get the probability.
		            double predictionProbability = 
		                predictionDistribution[predictionDistributionIndex];

		            System.out.printf("[%10s : %6.3f]", 
		                              predictionDistributionIndexAsClassLabel, 
		                              predictionProbability );
		        }

		        System.out.println("\n");
		    }
			
			
			
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			System.out.println(eval.toClassDetailsString());
			System.out.println("===== Evaluating on filtered (training) dataset done =====");
		}
		catch (Exception e) {
			System.out.println("Problem found when evaluating");
			System.out.println(e);
		}
	}

	/**
	 * This method trains the classifier on the loaded dataset.
	 */
	public void learn() {
		try {
			trainData.setClassIndex(1);
			filter = new StringToWordVector();
			filter.setAttributeIndices("first");
			classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			classifier.setClassifier(new RandomForest());
			classifier.buildClassifier(trainData);
			// Uncomment to see the classifier
			// System.out.println(classifier);
			System.out.println("===== Training on filtered (training) dataset done =====");
		}
		catch (Exception e) {
			System.out.println("Problem found when training");
		}
	}

	/**
	 * This method saves the trained model into a file. This is done by
	 * simple serialization of the classifier object.
	 * @param fileName The name of the file that will store the trained model.
	 */
	public void saveModel(String fileName) {
		try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(classifier);
            out.close();
 			System.out.println("===== Saved model: " + fileName + " =====");
        }
		catch (IOException e) {
			System.out.println("Problem found when writing: " + fileName);
		}
	}

	/**
	 * Main method. It is an example of the usage of this class.
	 * @param args Command-line arguments: fileData and fileModel.
	 */
	public static void main (String[] args) {

		MyFilteredLearner learner;
		if (args.length < 2)
			System.out.println("Usage: java MyLearner <fileTrainData> <fileTest> <fileModel>");
		else {
			learner = new MyFilteredLearner();
			learner.loadDataset(args[0]);
			learner.loadTestDataset(args[1]);
			
			learner.loadCSVData("imdb_norm.csv");
			// Evaluation must be done before training
			// More info in: http://weka.wikispaces.com/Use+WEKA+in+your+Java+code
			learner.evaluate();
			learner.learn();
			learner.saveModel(args[2]);
		}
	}
}
