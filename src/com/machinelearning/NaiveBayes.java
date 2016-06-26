package com.machinelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class NaiveBayes 
{
	private boolean useLaplace = false;
	
	private int numTrainingExamples = 0;
	private int numFeatures = 0;
	private int[][] featureMatrix;
	private int[] LabelVector;
	
	private int posCount = 0;
	private int negCount = 0;
	private int[] featureCountsPos;
	private int[] featureCountsNeg;
	
	public NaiveBayes(boolean useLap) {
		useLaplace = useLap;
	}
	
	private int getClassification(int[] featureVector)
	{
		double posDenom = posCount;
		double negDenom = negCount;
		
		if(useLaplace) {
			posDenom += featureCountsPos.length;
			negDenom += featureCountsNeg.length;
		}
		double logProbPos = Math.log((double) posCount / (posCount + negCount));
		double logProbNeg = Math.log((double) negCount / (posCount + negCount));

		double posClass = 0;
		double negClass = 0;
		for(int i = 0; i < featureVector.length; i++)
		{
			if(featureVector[i] == 1) {
				// has a "1" in position i and is of class pos or neg
				posClass = featureCountsPos[i];
				negClass = featureCountsNeg[i];
			} else {
				// has a "0" in position i and is of class positive or negative
				posClass = posCount - featureCountsPos[i];
				negClass = negCount - featureCountsNeg[i];
			}
			
			if(useLaplace) {
				posClass += 1;
				negClass += 1;
			}
			
			logProbPos += Math.log( posClass / posDenom );
			logProbNeg += Math.log( negClass / negDenom );
		}
		
		if(logProbPos > logProbNeg)
			return 1;
		
		return 0;
		
	}
	
	
	public void testNaiveBayes()
	{
		int numNeg = 0;
		int numPos = 0;
		int numCorrectNeg = 0;
		int numCorrectPos = 0;
		for(int i = 0; i < featureMatrix.length; i++)
		{
			int classification = getClassification(featureMatrix[i]);
			if(LabelVector[i] == 0) {
				numNeg++;
				if(classification == LabelVector[i])
					numCorrectNeg++;
			} else {
				numPos++;
				if(classification == LabelVector[i])
					numCorrectPos++;
			}
		}
		
		System.out.println("Class 0: tested " + numNeg + ", correctly classified " + numCorrectNeg);
		System.out.println("Class 1: tested " + numPos + ", correctly classified " + numCorrectPos);
		System.out.println("Overall: tested " + (numNeg+numPos) + ", correctly classified " + (numCorrectPos+numCorrectNeg));
		System.out.println("Accuracy = " + (double)(numCorrectPos+numCorrectNeg) / (numNeg+numPos));
		
	}
	
	
	/*
	 * Trains the naive bayes classification model by initializing an array
	 * for positive class counts and one for negative class counts.  It then
	 * iterates over the data and adds up the number of places where we see a
	 * 1 with Y = 1 and the number of places we see a 1 with Y = 0.
	 */
	public void trainNaiveBayes()
	{
		featureCountsPos = new int[numFeatures];
		featureCountsNeg = new int[numFeatures];
		
		for(int i = 0; i < featureMatrix.length; i++)
		{
			//Calculate the num of positive instance or negative instance
			if(LabelVector[i] == 1) 
				posCount++;
			else 
				negCount++;
			
			for(int j = 0; j < featureMatrix[0].length; j++)
			{
				if(LabelVector[i] == 1)
					featureCountsPos[j] += featureMatrix[i][j];
				else
					featureCountsNeg[j] += featureMatrix[i][j];
			}
		}
	}
	
	/*
	 * Input file are always of the format
	 *    <number of features>
	 *    <number of training examples>
	 *    < ... data ... >
	 *    <feature data> : <label>
	 * This method reads those constants and sets up the appropriate instance variables.
	 */
	private void readFileConstants(BufferedReader input) throws NumberFormatException, IOException
	{
		// Get num features and num training examples
		numFeatures = Integer.parseInt(input.readLine());
		numTrainingExamples = Integer.parseInt(input.readLine());
		
		featureMatrix = new int[numTrainingExamples][numFeatures];
		LabelVector = new int[numTrainingExamples];
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

				lineVector = line.split(" ");
				for(int j = 0; j < lineVector.length - 1; j++)
				{
					// semi-colon denotes the end of the feature data
					if(lineVector[j].indexOf(':') != -1) {
						lineVector[j] = lineVector[j].substring(0, 1);
					}
					featureMatrix[i][j] = Integer.parseInt(lineVector[j]);
				}
				//The last position of line is "Label"
				LabelVector[i] = Integer.parseInt(lineVector[lineVector.length-1]);
				i++;
			}
			input.close();
			
		} catch(IOException e) {
			e.printStackTrace();
			System.exit(1);
		} 
	}
	
}
