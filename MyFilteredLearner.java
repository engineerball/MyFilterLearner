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
import weka.core.stemmers.LovinsStemmer;

import java.io.*;
import org.apache.commons.lang3.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import java.util.Set;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

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
	
	
	HashMap<String, Double> positiveBC = new HashMap<String, Double>();
	HashMap<String, Double> positiveDC = new HashMap<String, Double>();
	HashMap<String, Double> positiveCC = new HashMap<String, Double>();
	HashMap<String, Double> negativeBC = new HashMap<String, Double>();
	HashMap<String, Double> negativeDC = new HashMap<String, Double>();
	HashMap<String, Double> negativeCC = new HashMap<String, Double>();
	
	Map<Integer, List<String>> positiveFoundWord = new HashMap<Integer, List<String>>();
	HashMap<Integer, Double> positiveFoundWordBC = new HashMap<Integer, Double>();
	HashMap<Integer, Double> positiveFoundWordDC = new HashMap<Integer, Double>();
	HashMap<Integer, Double> positiveFoundWordCC = new HashMap<Integer, Double>();
	Map<Integer, List<String>> negativeFoundWord = new HashMap<Integer, List<String>>();
	HashMap<Integer, Double> negativeFoundWordBC = new HashMap<Integer, Double>();
	HashMap<Integer, Double> negativeFoundWordDC = new HashMap<Integer, Double>();
	HashMap<Integer, Double> negativeFoundWordCC = new HashMap<Integer, Double>();
	
	Map<Integer, String> classLabel = new HashMap<Integer, String>();
	Map<Integer, String> predictedClass = new HashMap<Integer, String>();
	Map<Integer, Double> distributionProb0 = new HashMap<Integer, Double>();
	Map<Integer, Double> distributionProb1 = new HashMap<Integer, Double>();

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

	
	public void loadPositiveCSVData(String fileName) {
		String line = "";
        String cvsSplitBy = ",";

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {

            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] data = line.split(cvsSplitBy);

//                System.out.println("Data [code= " + data[0] + " , name=" + data[1] + " , BC=" + data[2] + " , CC=" + data[3] + " , DC=" + data[4] + "]");

                positiveBC.put(data[1].toString(), Double.parseDouble(data[2]));
                positiveCC.put(data[1].toString(), Double.parseDouble(data[3]));
                positiveDC.put(data[1].toString(), Double.parseDouble(data[4]));
                
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
		
	}

	public void loadNegativeCSVData(String fileName) {
		String line = "";
        String cvsSplitBy = ",";

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {

            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] data = line.split(cvsSplitBy);

//                System.out.println("Data [code= " + data[0] + " , name=" + data[1] + " , BC=" + data[2] + " , CC=" + data[3] + " , DC=" + data[4] + "]");

                negativeBC.put(data[1].toString(), Double.parseDouble(data[2]));
                negativeCC.put(data[1].toString(), Double.parseDouble(data[3]));
                negativeDC.put(data[1].toString(), Double.parseDouble(data[4]));
               
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
		
	}

	public String searchString(String classString, String word) {
		String[] s1 = classString.split("[ .,;'\"!]+");
		String sLower;
		String foundword = "";
		for (String s : s1){
			sLower = s.toLowerCase();
//			System.out.println(sLower);
			if (word.equals(sLower)) {				
				foundword = word;
//				System.out.println("Found :" + foundword);
			} 			
		}
		return foundword;
	}
	
	
	public Double sumCentrality(List words, HashMap<String, Double> centralityMap) {
		Double sum = 0.0;
		for ( int i = 0 ; i < words.size(); i++){
			sum += centralityMap.get(words.get(i)); 
		}		
		return sum;
	}
	
	public void writePosCentralityToCSV(HashMap<Integer, Double> BCMap,HashMap<Integer, Double> CCMap,HashMap<Integer, Double> DCMap) {
		PrintWriter pw = null;
		try {
		    pw = new PrintWriter(new File("Pos_Centrality.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "Id,BC,CC,DC";
		builder.append(ColumnNamesList +"\n");
		
		for (int i = 0 ; i < BCMap.size(); i++){
			builder.append(i + "," + BCMap.get(i) + "," + CCMap.get(i) + "," + DCMap.get(i));
			builder.append('\n');

		}
		pw.write(builder.toString());		
		pw.close();
		System.out.println("done!");
	}

	public void writeNegCentralityToCSV(HashMap<Integer, Double> BCMap,HashMap<Integer, Double> CCMap,HashMap<Integer, Double> DCMap) {
		PrintWriter pw = null;
		try {
		    pw = new PrintWriter(new File("Neg_Centrality.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "Id,BC,CC,DC";
		builder.append(ColumnNamesList +"\n");
		
		for (int i = 0 ; i < BCMap.size(); i++){
			builder.append(i + "," + BCMap.get(i) + "," + CCMap.get(i) + "," + DCMap.get(i));
			builder.append('\n');

		}
		pw.write(builder.toString());		
		pw.close();
		System.out.println("done!");
	}

	public void writeDistributedProb0ToCSV(Map<Integer, Double> distributionProb){
		PrintWriter pw = null;
		
		try {
		    pw = new PrintWriter(new File("distribution0prob.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "ID, prob";
		builder.append(ColumnNamesList +"\n");
		
		for (int i = 0 ; i < distributionProb.size(); i++){
			builder.append(i + "," + distributionProb.get(i));
			builder.append('\n');

		}
		pw.write(builder.toString());		
		pw.close();		
	}

	public void writeDistributedProb1ToCSV(Map<Integer, Double> distributionProb){
		PrintWriter pw = null;
		
		try {
		    pw = new PrintWriter(new File("distribution1prob.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "ID, prob";
		builder.append(ColumnNamesList +"\n");
		
		for (int i = 0 ; i < distributionProb.size(); i++){
			builder.append(i + "," + distributionProb.get(i));
			builder.append('\n');

		}
		pw.write(builder.toString());		
		pw.close();		
	}
	
	public void writePositiveFoundWordsToText(Map<Integer, List<String>> positiveFoundWord) {
		String pathToFile = "positiveFoundWords.txt";
		try {
			FileWriter writer;
			writer = new FileWriter(pathToFile, true);
			for (int i = 0; i < positiveFoundWord.size(); i++) {
//				writer.write(i);;
//				writer.write(" : ");
				writer.write(positiveFoundWord.get(i).toString());
				writer.write("\r\n");
			}
			System.out.println("Done");
			writer.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public void writeNegativeFoundWordsToText(Map<Integer, List<String>> negativeFoundWord) {
		String pathToFile = "negativeFoundWords.txt";
		try {
			FileWriter writer;
			writer = new FileWriter(pathToFile, true);
			for (int i = 0; i < negativeFoundWord.size(); i++) {
//				writer.write(i);;
//				writer.write(" : ");
				writer.write(negativeFoundWord.get(i).toString());
				writer.write("\r\n");
			}
			System.out.println("Done");
			writer.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public void writePredictedClassToCSV(Map<Integer, String> labelClass, Map<Integer, String> predictedClass) {
		String pathToFile = "predictedClass.csv";
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new File(pathToFile));
		} catch (Exception e) {
			e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "ID,label,predicted";
		builder.append(ColumnNamesList + "\n");
		for (int i = 0; i < labelClass.size(); i++) {
			builder.append(i + "," + labelClass.get(i) + "," + predictedClass.get(i));
			builder.append('\n');
		}
		pw.write(builder.toString());
		pw.close();
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
			filter.setStemmer(new LovinsStemmer());
			filter.setAttributeIndices("first");
			System.out.println("Filter options : " + Arrays.toString(filter.getOptions()));
			classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			
			RandomForest rf = new RandomForest();
			rf.setSeed(2);
			rf.setNumTrees(100);
			rf.setMaxDepth(0);
			rf.setNumFeatures(0);
			System.out.println("RandomForest options : " + Arrays.toString(rf.getOptions()));
			
			classifier.setClassifier(rf);
			classifier.buildClassifier(trainData);
//			System.out.println(testData);
			
			String searchResult = "";
			List<String> posFoundWords = new ArrayList<String>();
			List<String> negFoundWords = new ArrayList<String>();
			
			Double posBC, posDC, posCC, negBC, negDC, negCC = 0.0;
			
			for ( int i = 0 ; i < testData.numInstances(); i++)
			{

//				positiveFoundWord.clear();
				String stringClass = testData.instance(i).toString(testData.classIndex() - 1);
//				System.out.println(stringClass);
				posFoundWords.clear();
				if (positiveFoundWord.get(i) == null) {
			    	positiveFoundWord.put(i, new ArrayList<String>());
					
			    	Set posSet = positiveBC.entrySet();
				    Iterator posIterator = posSet.iterator();
				    while(posIterator.hasNext()) {
				       Map.Entry mentry = (Map.Entry)posIterator.next();		
				       searchResult = searchString(stringClass, mentry.getKey().toString());
				       if (searchResult != ""){
				    	   positiveFoundWord.get(i).add(searchResult);
				    	   posFoundWords.add(searchResult);
				       }
				    }
				}
			    

			    posBC = sumCentrality(posFoundWords, positiveBC);
			    posDC = sumCentrality(posFoundWords, positiveDC);
			    posCC = sumCentrality(posFoundWords, positiveCC);
			    			    
			    			    
			    positiveFoundWordBC.put(i, posBC);
			    positiveFoundWordDC.put(i, posDC);
			    positiveFoundWordCC.put(i, posCC);
			    
			    /* Negative method */
			    negFoundWords.clear();
			    if (negativeFoundWord.get(i) == null) {
			    	negativeFoundWord.put(i,  new ArrayList<String>());
				    Set negSet = negativeBC.entrySet();
				    Iterator negIterator = negSet.iterator();
				    while(negIterator.hasNext()) {
				       Map.Entry mentry = (Map.Entry)negIterator.next();		
				       searchResult = searchString(stringClass, mentry.getKey().toString());
				       if (searchResult != ""){
				    	   negativeFoundWord.get(i).add(searchResult);
				    	   negFoundWords.add(searchResult);
				       }
				    }
			    }
			    
			    negBC = sumCentrality(negFoundWords, negativeBC);
			    negDC = sumCentrality(negFoundWords, negativeDC);
			    negCC = sumCentrality(negFoundWords, negativeCC);
		    			    
			    negativeFoundWordBC.put(i, negBC);
			    negativeFoundWordDC.put(i, negDC);
			    negativeFoundWordCC.put(i, negCC);
			}
			
			
//			System.out.println(positiveFoundWordBC);
//			System.out.println(positiveFoundWordDC);
//			System.out.println(positiveFoundWordCC);
//			
//			System.out.println(negativeFoundWordBC);
//			System.out.println(negativeFoundWordBC);
//			System.out.println(negativeFoundWordBC);
			
//			System.out.println(positiveFoundWord);
			writePositiveFoundWordsToText(positiveFoundWord);
			writeNegativeFoundWordsToText(negativeFoundWord);
//			System.out.println(negativeFoundWord);
			
			writePosCentralityToCSV(positiveFoundWordBC, positiveFoundWordCC, positiveFoundWordDC);
			writeNegCentralityToCSV(negativeFoundWordBC, negativeFoundWordCC, negativeFoundWordDC);
			
			
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
		        System.out.printf("%5d: true=%-10s, predicted=%-10s, distribution=", i, trueClassLabel, predictedClassLabel); 
//		        System.out.printf("%5d;", i);
		        
		        // Stored in HashMap
		        classLabel.put(i, trueClassLabel.toString());
		        predictedClass.put(i, predictedClassLabel.toString());
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

		            System.out.printf("[%10s : %6.3f]", predictionDistributionIndexAsClassLabel, predictionProbability );
//		            System.out.printf("%6.3f;", predictionProbability);
		            if (predictionDistributionIndexAsClassLabel.equals("0")) {
		            	distributionProb0.put(i, predictionProbability);
		            } else {
		            	distributionProb1.put(i, predictionProbability);
		            }
		        }

		        System.out.println("\n");
		    }
		    writeDistributedProb0ToCSV(distributionProb0);
		    writeDistributedProb1ToCSV(distributionProb1);
		    writePredictedClassToCSV(classLabel, predictedClass);
//			System.out.println(distributionProb0);
//			System.out.println(distributionProb1);
			
			
			
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
			System.out.println("Usage: java MyLearner <fileTrainData> <fileTest>  <positive_wordlist.csv> <negative_wordlist.csv> <fileModel>");
		else {
			learner = new MyFilteredLearner();
			learner.loadDataset(args[0]);
			learner.loadTestDataset(args[1]);
			
			learner.loadPositiveCSVData(args[2]);
			learner.loadNegativeCSVData(args[3]);
			// Evaluation must be done before training
			// More info in: http://weka.wikispaces.com/Use+WEKA+in+your+Java+code
			learner.evaluate();	      
			learner.learn();
			learner.saveModel(args[4]);
		}
	}
}
