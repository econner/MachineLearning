package com.machinelearning;

public class MachineLearning {
	// PARAMETERS
	private String dataDir = "datasets/";
	private String trainFile = "heart-train.txt";
	private String testFile = "heart-test.txt";

	public void doNaiveBayes() {

		// PARAMETER
		// use laplace or not?
		boolean useLaplace = true;

		NaiveBayes nb = new NaiveBayes(useLaplace);
		nb.readFeatureData(dataDir + trainFile);
		nb.trainNaiveBayes();

		nb.readFeatureData(dataDir + testFile);
		nb.testNaiveBayes();
	}

	public void doLogisticRegression() {
		LogisticRegression lr = new LogisticRegression();
		lr.readFeatureData(dataDir + trainFile);
		lr.trainLogisticRegression();

		lr.readFeatureData(dataDir + testFile);
		lr.testLogisticRegression();
	}

	public static void main(String[] args) {
		MachineLearning ml = new MachineLearning();
		// ml.doNaiveBayes();
		ml.doLogisticRegression();
	}
}
