package com.machinelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class LogisticRegression {
	private static final int NUM_EPOCHS = 10000;
	private static final double LEARNING_RATE = 0.000001;
	
	private int numTrainingExamples = 0;
	private int numFeatures = 0;
	private int[][] featureMatrix;
	private int[] yVector;
	
	// the learned weights
	private double[] weights;
	
	/*
	 * Tests the logistic regression algorithm, assuming
	 * the data has been correctly loaded into the instance variables
	 * defined above.
	 */
	public void testLogisticRegression()
	{
		int numNeg = 0;
		int numPos = 0;
		int numCorrectNeg = 0;
		int numCorrectPos = 0;
		for(int i = 0; i < featureMatrix.length; i++)
		{
			int classification = getClassification(featureMatrix[i]);
			if(yVector[i] == 0) {
				numNeg++;
				if(classification == yVector[i])
					numCorrectNeg++;
			} else {
				numPos++;
				if(classification == yVector[i])
					numCorrectPos++;
			}
		}
		
		System.out.println("Class 0: tested " + numNeg + ", correctly classified " + numCorrectNeg);
		System.out.println("Class 1: tested " + numPos + ", correctly classified " + numCorrectPos);
		System.out.println("Overall: tested " + (numNeg+numPos) + ", correctly classified " + (numCorrectPos+numCorrectNeg));
		System.out.println("Accuracy = " + (double)(numCorrectPos+numCorrectNeg) / (numNeg+numPos));
	}
	
	/*
	 * Calculates the linear term in the sigmoid function ("z")
	 */
	private double calculateLinearTerm(int[] featureVector)
	{
		double linearTerm = 0;
		for(int i = 0; i < featureVector.length; i++)
		{
			linearTerm += weights[i] * (double)featureVector[i];
		}
		return linearTerm;
	}
	
	/*
	 * Gets our binary classification for a given feature vector
	 */
	private int getClassification(int[] featureVector) {
		double logOdds = calculateLinearTerm(featureVector);
		
		if(logOdds > 0)
			return 1;
		return 0;
	}
	
	/*
	 * Trains logistic regression using batch
	 * gradient descent
	 */
	public void trainLogisticRegression()
	{
		// Initialize: weights = 0 for all 0 ² j² m
		// add 1 for bias term
		weights = new double[numFeatures+1];
		for(int i = 0; i < weights.length; i++) weights[i] = 0;
		
		for(int i = 0; i < NUM_EPOCHS; i++)
		{
			// add 1 for bias term
			double[] gradient = new double[numFeatures+1];
			for(int g = 0; g < gradient.length; g++) gradient[g] = 0;
			
			for(int row = 0; row < featureMatrix.length; row++)
			{
				for(int col = 0; col < featureMatrix[0].length; col++)
				{
					gradient[col] += (double)featureMatrix[row][col]*((double)yVector[row] - sigmoid(featureMatrix[row]));
				}
			}
			
			for(int j = 0; j < weights.length; j++)
				weights[j] += LEARNING_RATE * gradient[j];
		}
	}
	

	/*
	 * Calculates the sigmoid function (of the form 1 / (1 + e^(-z))
	 */
	private double sigmoid(int[] featureVector) {
		
		double linearTerm = calculateLinearTerm(featureVector);
		return 1.0 / (1.0 + Math.exp(-linearTerm));
	}

	/*
	 * Input file are always of the format
	 *    <number of features>
	 *    <number of training examples>
	 *    < ... data ... >
	 * This method reads those constants and sets up the appropriate instance variables.
	 */
	private void readFileConstants(BufferedReader input) throws NumberFormatException, IOException
	{
		// Get num features and num training examples
		numFeatures = Integer.parseInt(input.readLine());
		numTrainingExamples = Integer.parseInt(input.readLine());
		
		// add 1 for bias term
		featureMatrix = new int[numTrainingExamples][numFeatures+1];
		yVector = new int[numTrainingExamples];
		
	}
	
	/*
	 * Reads in the feature data and ground truth vector from 
	 * given input file.
	 */
	public void readFeatureData(String fname)
	{
		try {

			BufferedReader input = new BufferedReader(new FileReader(fname));
			readFileConstants(input);
			
			String[] lineVector;
			int i = 0;
			for(String line = input.readLine(); line != null; line = input.readLine()) {
				// bias term
				featureMatrix[i][0] = 1;
				
				lineVector = line.split(" ");
				for(int j = 0; j < lineVector.length - 1; j++)
				{
					// semi-colon denotes the end of the feature data
					if(lineVector[j].indexOf(':') != -1) {
						lineVector[j] = lineVector[j].substring(0, 1);
					}
					featureMatrix[i][j+1] = Integer.parseInt(lineVector[j]);
				}
				yVector[i] = Integer.parseInt(lineVector[lineVector.length-1]);

				i++;
			}
			input.close();
			
			//printFeatureMatrix();
			
		} catch(IOException e) {
			e.printStackTrace();
			System.exit(1);
		} 
	}
}
